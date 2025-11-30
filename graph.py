####################################################################################################
# PROGRAM NAME: Research Agent Workflow (Graph.py)
# --------------------------------------------------------------------------------------------------
# DESCRIPTION: Defines the LangGraph state machine for an autonomous research assistant.
# It orchestrates a "Diamond Pattern" workflow:
# 1. Divergence: Intent detection routes to a Primary Tool.
# 2. Parallelism/Sequence: Primary Tool executes -> Secondary Tools execute.
# 3. Convergence: All data aggregates into a Vector Store for semantic ranking.
# 4. Synthesis: LLM generates a final report with trend analysis.
#
# AUTHOR: AI Agent Team
# DATE: 2025-11-26
# VERSION: 7.2 (Final Debug - Reference Filter Applied)
####################################################################################################

import os
import sys
import io
import time
import json
import pickle
import traceback
import ast
import re
import numpy as np
import faiss
import arxiv
import requests
import traceback
from typing import TypedDict, Any, Dict, List, Optional, Annotated
from dotenv import load_dotenv

# Third-party integrations
from pypdf import PdfReader
from openai import OpenAI
from mp_api.client import MPRester
from langgraph.graph import StateGraph, END
from graphviz import Digraph
from Bio import Entrez
from ddgs import DDGS
from datetime import datetime


# ------------------------------------------------------------------------------
# ANSI Color Codes for CLI Debugging
# ------------------------------------------------------------------------------
C_RESET = "\033[0m"
C_RED = "\033[91m"      # Errors/Fatal
C_GREEN = "\033[92m"    # Success/Init
C_YELLOW = "\033[93m"   # Warnings/Skips
C_BLUE = "\033[94m"     # Routes/Actions
C_MAGENTA = "\033[95m"  # Step Headers
C_CYAN = "\033[96m"     # Plans/Info
C_ACTION = "\033[38;5;208m"  # Example for a bright orange/action color

# ==================================================================================================
# SECTION 1: CONFIGURATION & ENVIRONMENT
# ==================================================================================================
print(f"{C_GREEN} >> [INIT] Loading environment variables...{C_RESET}")
load_dotenv()

OPENAI_API_KEY = os.getenv("GPT_API_KEY")
MP_API_KEY = os.getenv("MP_API_KEY")
EMBED_MODEL = "text-embedding-3-small"


if not OPENAI_API_KEY:
    print(f"{C_RED} >> [FATAL] GPT_API_KEY not found in environment. Aborting.{C_RESET}")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)
print(f"{C_GREEN} >> [INIT] OpenAI client initialized successfully.{C_RESET}")

# Centralized Tool Mapping
TOOL_MAPPING = {
    "arxiv_search": "tool_arxiv",
    "materials_project": "tool_mp",
    "vector_search": "tool_vector",
    "python_compute": "tool_python",
    "openalex_search": "tool_openalex",
    "pubmed_search": "tool_pubmed",
    "paper_retrieve": "tool_full_text_expander",
    "patent_search": "tool_patent_search",
    "google_search": "tool_google_search",
    "unknown": "tool_arxiv"
}

# ==================================================================================================
# SECTION 2: STATE DEFINITIONS
# ==================================================================================================
class ResearchState(TypedDict, total=False):
    query: str
    api_search_term: str
    semantic_query: str
    clean_query: str
    intent: str
    secondary_tools: List[str]
    tool_reasoning: Dict[str, str]
    material_elements: List[str]
    tool: str
    raw_data: Any
    multi_tool_data: Annotated[Dict[str, Any], "Concurrent tool outputs"]
    references: List[str]
    filtered_context: str
    tool_result: str
    insights_list: List[str]
    answer: str
    cached_results: Dict[str, Any]


# ==================================================================================================
# SECTION 3: PERSISTENCE LAYER
# ==================================================================================================
DIMENSION = 1536
VECTOR_INDEX_PATH = "vector_index.faiss"
VECTOR_DATA_PATH = "vector_data.pkl"

print(f"{C_GREEN} >> [INIT] Checking Vector DB at {VECTOR_INDEX_PATH}...{C_RESET}")
if os.path.exists(VECTOR_INDEX_PATH) and os.path.exists(VECTOR_DATA_PATH):
    persistent_index = faiss.read_index(VECTOR_INDEX_PATH)
    with open(VECTOR_DATA_PATH, "rb") as f:
        persistent_db_vectors = pickle.load(f)
    print(f"{C_GREEN} >> [INIT] FAISS vector DB loaded from disk.{C_RESET}")
else:
    index = faiss.IndexFlatL2(DIMENSION)
    faiss.write_index(index, VECTOR_INDEX_PATH)
    with open(VECTOR_DATA_PATH, "wb") as f:
        pickle.dump(np.array([]).astype("float32"), f)
    print(f"{C_GREEN} >> [INIT] FAISS vector DB initialized as EMPTY structure and saved to disk.{C_RESET}")


# ==================================================================================================
# SECTION 4: CORE LOGIC NODES
# ==================================================================================================

def clean_query(state: ResearchState) -> ResearchState:
    query = state.get('query', '')
    print(f"{C_MAGENTA}\n[STEP 1] clean_query{C_RESET} | Input: '{query}'")
    if not query:
        print(f"{C_RED} >> [FATAL] Original query is empty. Setting fallbacks to empty strings.{C_RESET}")
        state["api_search_term"] = ""
        state["semantic_query"] = ""
        state["clean_query"] = ""
        return state

    prompt_api = f"Analyze the user query: '{query}'. Extract the single most relevant search term/material. Respond ONLY with the term."
    try:
        resp_api = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt_api}], max_tokens=50, temperature=0
        )
        api_term = resp_api.choices[0].message.content.strip()
        state["api_search_term"] = api_term
        print(f"{C_GREEN} >> [SUCCESS] API Search Term: '{api_term}'{C_RESET}")
    except Exception as e:
        print(f"{C_RED} >> [ERROR] API Term generation failed: {e}. Falling back to full query.{C_RESET}")
        state["api_search_term"] = query

    prompt_semantic = f"Analyze the user query: '{query}'. Combine all core concepts, intent, and subject into a single, semantically rich phrase separated by commas. Respond ONLY with the phrase."
    try:
        resp_sem = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt_semantic}], max_tokens=70, temperature=0
        )
        sem_query = resp_sem.choices[0].message.content.strip()
        state["semantic_query"] = sem_query
        state["clean_query"] = sem_query
        print(f"{C_GREEN} >> [SUCCESS] Semantic Query: '{sem_query}'{C_RESET}")
    except Exception as e:
        print(f"{C_RED} >> [ERROR] Semantic Query generation failed: {e}. Falling back to API term.{C_RESET}")
        state["semantic_query"] = state["api_search_term"]

    return state


def detect_intent(state: ResearchState) -> ResearchState:
    query = state.get('clean_query', state.get('query', ''))
    print(f"{C_MAGENTA}[STEP 2] detect_intent{C_RESET} | Planning research for: '{query}'")

    prompt = f"""
    You are a research planner.
    Available Tools: {list(TOOL_MAPPING.keys())}

    Task:
    1. Select ONE Primary Tool (most relevant).
    2. Select ANY Secondary Tools (for cross-referencing).
    3. If the query concerns a chemical compound, extract ALL valid chemical symbols (e.g., ['Si', 'O']).

    Respond ONLY in the following JSON format:
    {{
        "primary": "tool_name",
        "secondary": ["tool_a", "tool_b"],
        "material_elements": ["Si", "O"],
        "reasoning": {{"tool_name": "reason"}}
    }}

    Query: "{query}"
    """

    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0,
                response_format={"type": "json_object"}
            )
            raw_content = resp.choices[0].message.content.strip()

            try:
                data = json.loads(raw_content)
            except json.JSONDecodeError:
                print(f"{C_YELLOW} >> [WARN] JSON load failed. Attempting regex recovery...{C_RESET}")
                match = re.search(r'(\{.*?})\s*$', raw_content, re.DOTALL)
                if match:
                    data = ast.literal_eval(match.group(1))
                else:
                    raise ValueError("Could not find a valid JSON object in the response.")

            primary_intent = data.get("primary", "arxiv_search")
            secondary_tools = data.get("secondary", [])

            # --- 🌟 ENFORCEMENT RULES FOR GUARANTEED EXECUTION 🌟 ---
            academic_tools = ["arxiv_search", "pubmed_search"]
            if primary_intent in academic_tools or any(tool in secondary_tools for tool in academic_tools):
                if "paper_retrieve" not in secondary_tools:
                    secondary_tools.append("paper_retrieve")
                    print(f"{C_BLUE} >> [ENFORCE] Added 'paper_retrieve' for full-text RAG.{C_RESET}")

                if "pubmed_search" in [primary_intent] + secondary_tools and "pubmed_metadata_lookup" not in secondary_tools:
                    secondary_tools.append("pubmed_metadata_lookup")
                    print(f"{C_BLUE} >> [ENFORCE] Added 'pubmed_metadata_lookup' due to PubMed dependency.{C_RESET}")

            if "google_search" not in secondary_tools:
                secondary_tools.append("google_search")
                print(f"{C_BLUE} >> [ENFORCE] Added 'google_search' as general safety net.{C_RESET}")
            # ---------------------------------------------------------

            state["intent"] = primary_intent
            state["secondary_tools"] = secondary_tools
            state["tool_reasoning"] = data.get("reasoning", {})

            elements = data.get("material_elements", [])
            state["material_elements"] = [e for e in elements if isinstance(e, str) and e]

            state.setdefault("multi_tool_data", {})

            print(f"{C_CYAN} >> [PLAN] Primary Intent: {state['intent']}{C_RESET}")
            print(f"{C_CYAN} >> [PLAN] Secondary Tools: {state['secondary_tools']}{C_RESET}")
            if state["material_elements"]:
                 print(f"{C_CYAN} >> [PLAN] Extracted Elements: {state['material_elements']}{C_RESET}")

            return state

        except Exception as e:
            print(f"{C_RED} >> [ERROR] Intent detection failed on attempt {attempt + 1}: {e}{C_RESET}")
            if attempt == 1:
                print(f"{C_RED} >> [FATAL FALLBACK] All attempts failed. Defaulting to general search.{C_RESET}")
                state["intent"] = "google_search"
                state["secondary_tools"] = ["pubmed_search", "pubmed_metadata_lookup", "arxiv_search"]
                state["material_elements"] = []
                state.setdefault("multi_tool_data", {})
                return state

            time.sleep(0.5)

    return state


def route_to_tool(state: ResearchState) -> ResearchState:
    print(f"{C_MAGENTA}[STEP 3] route_to_tool{C_RESET} | Calculating path...")
    intent = state.get("intent", "arxiv_search")
    target = TOOL_MAPPING.get(intent, "tool_arxiv")
    state["tool"] = target
    print(f"{C_BLUE} >> [ROUTE] Directed to node: {target}{C_RESET}")
    return state



################################################################################
# HELPER: _granular_chunker (FINAL CORRECTED VERSION)
################################################################################
def _granular_chunker(multi_tool_data: Dict[str, Any]) -> List[str]:
    """Flattens and chunks data from tool outputs into meaningful, embeddable units."""
    text_blocks = []

    # New function to process and cleanse a single chunk
    def _process_and_cleanse_chunk(chunk: str) -> str:
        # 1. Strip common whitespace issues
        clean_chunk = chunk.strip()
        # 2. CRITICAL: Force encode/decode to strip all non-safe Unicode/surrogates
        try:
            return clean_chunk.encode('ascii', 'ignore').decode('utf-8')
        except Exception:
            return clean_chunk

    for tool_name, data in multi_tool_data.items():
        if tool_name == "vector_search": continue

        if not data or (isinstance(data, dict) and "error" in data):
            print(f"{C_YELLOW}    > Skipping {tool_name}: Data is empty or erroneous.{C_RESET}")
            continue

        print(f"{C_BLUE}    > Chunking data from {tool_name}{C_RESET}")

        if tool_name == "arxiv_search" and isinstance(data, list):
            for i, item in enumerate(data):
                title = item.get("title", "No Title")
                summary = item.get("summary", "No Summary")
                block = f"Source: ArXiv Paper {i+1}. Title: {title}. Summary: {summary}"
                text_blocks.append(_process_and_cleanse_chunk(block))

        elif tool_name == "materials_project" and isinstance(data, list):
            for i, item in enumerate(data):
                mid = item.get("id", "N/A")
                formula = item.get("formula", "N/A")
                stability = "Stable" if item.get("stability", False) else "Unstable"
                block = f"Source: Materials Project ID {mid}. Formula: {formula}. Stability: {stability}."
                text_blocks.append(_process_and_cleanse_chunk(block))

        elif tool_name == "full_text_expander" and isinstance(data, list):
            # full_text_expander chunks are already formatted strings
            cleansed_chunks = [_process_and_cleanse_chunk(c) for c in data]
            text_blocks.extend(cleansed_chunks)
            print(f"{C_GREEN}    > Added {len(cleansed_chunks)} full-text chunks.{C_RESET}")

        elif tool_name == "google_search" and isinstance(data, list):
            # google_search chunks are already formatted strings (from _call_ddg_search)
            cleansed_chunks = [_process_and_cleanse_chunk(c) for c in data]
            text_blocks.extend(cleansed_chunks)
            print(f"{C_GREEN}    > Added {len(cleansed_chunks)} general web chunks.{C_RESET}")

        elif isinstance(data, list):
            # This handles pubmed_search, openalex_search, patent_search
            for i, item in enumerate(data):
                block = f"Source: {tool_name.replace('_search', '').capitalize()} Entry {i+1}. Data: {json.dumps(item)[:150]}"
                text_blocks.append(_process_and_cleanse_chunk(block))

        else:
            # This handles generic single dictionary outputs
            block = f"Source: {tool_name.replace('_search', '').capitalize()} Data. Content: {json.dumps(data)[:800]}"
            text_blocks.append(_process_and_cleanse_chunk(block))

    return text_blocks


# ==================================================================================================
# SECTION 5: TOOL IMPLEMENTATIONS (Primary and Secondary)
# ==================================================================================================

def _extract_target_year(query: str) -> int:
    """Extracts a target year (four digits >= 2000) from the user query."""
    # Use the current year as the default maximum for recency checks
    current_max_year = datetime.now().year + 1

    match = re.search(r'\b(20[0-9]{2})\b', query)
    if match:
        extracted_year = int(match.group(1))
        # Ensure the extracted year is not far in the future
        if extracted_year <= current_max_year:
            return extracted_year
    # Default to current year if no valid year is found
    return datetime.now().year

def _construct_strict_boolean_query(semantic_query: str) -> str | None:
    """
    Constructs a strict Boolean query if high specificity is detected (Technology, App, Method),
    otherwise returns None. This function should be defined/imported globally.
    """
    query_lower = semantic_query.lower()

    # Define categories of keywords for specificity check
    tech_terms = ["aptamer", "biosensor", "detection", "sensor"]
    app_terms = ["cancer", "tumor", "diagnosis", "disease"]
    method_terms = ["machine learning", "ai", "artificial intelligence", "ml"]

    # Check for presence in all three categories
    has_tech = any(term in query_lower for term in tech_terms)
    has_app = any(term in query_lower for term in app_terms)
    has_method = any(term in query_lower for term in method_terms)

    if has_tech and has_app and has_method:
        print(f"{C_BLUE} >> [INFO] Strict Boolean search criteria met.{C_RESET}")

        # Core Tech (Must be present)
        core_tech = '("aptamer" OR "biosensor" OR "sensor")'

        # Target/Application (Must be present)
        target_app = '("cancer" OR "tumor" OR "diagnosis" OR "disease")'

        # Methodology (Must be present)
        methodology = '("machine learning" OR "AI" OR "ML" OR "artificial intelligence")'

        # Final query enforcing AND for all concepts and OR for synonyms
        strict_query = f'{core_tech} AND {target_app} AND {methodology}'
        return strict_query

    return None # Return None if not highly specific

def tool_arxiv(state: ResearchState) -> ResearchState: # Changed from Dict[str, Any] to ResearchState if available
    print(f"\n{C_MAGENTA}--------------------------------------------------{C_RESET}")
    print(f"{C_MAGENTA}[STEP 4.ArXiv] ARXIV SEARCH START{C_RESET} | Executing Search with Optimized Priority and Filters...{C_RESET}")
    print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")

    multi = state.setdefault("multi_tool_data", {})
    refs = state.setdefault("references", [])

    complex_query = state.get("semantic_query", "")
    simple_query = state.get("api_search_term", "")

    if not complex_query or not simple_query:
        print(f"{C_RED} >> [ERROR] Tool skipping execution due to missing search terms.{C_RESET}")
        multi["arxiv_search"] = {"error": "Missing search query provided."}
        return state

    # --- 1. Date Range Calculation (Recency Filter) ---
    current_year = datetime.now().year
    date_filter = f"AND (({current_year-1}:*) [submittedDate])"
    print(f"{C_CYAN} >> [INFO] Applying ArXiv Date Filter: Papers since {current_year-1}...{C_RESET}")

    # --- 2. Define Optimized Search Attempts ---
    query_attempts = [] # Stores (query_string, attempt_name) tuples
    strict_boolean_query = _construct_strict_boolean_query(complex_query)

    # Priority 1: STRICT BOOLEAN (Highest Relevance, Any Date)
    if strict_boolean_query:
        query_attempts.append((strict_boolean_query, "STRICT BOOLEAN (Any Date)"))

    # Priority 2: STRICT BOOLEAN + DATE FILTER (High Relevance, Recent)
    if strict_boolean_query:
        query_attempts.append((f"{strict_boolean_query} {date_filter}", "STRICT BOOLEAN + DATE"))

    # Priority 3: COMPLEX SEMANTIC + DATE FILTER (Medium Relevance, Recent)
    query_attempts.append((f"{complex_query} {date_filter}", "COMPLEX SEMANTIC + DATE"))

    # Priority 4: SIMPLE LITERAL (Lowest Relevance, Absolute Fallback)
    if simple_query:
        query_attempts.append((simple_query, "SIMPLE LITERAL (Fallback)"))


    # --- Search Executor Helper Function (Remains the same) ---
    def _execute_arxiv_search(term, max_count=3):
        results = []
        print(f"{C_CYAN} >> [ACTION] ArXiv searching for: {term[:80]}...{C_RESET}")

        search = arxiv.Search(
            query=term,
            max_results=max_count * 2,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        for r in arxiv.Client().results(search):
            results.append(r)
            if len(results) >= max_count:
                break
        return results

    # --- 3. Search Execution with Optimized Fallback ---
    results_filtered = []

    for i, (current_query, attempt_name) in enumerate(query_attempts): # Unpack the tuple here

        try:
            results_filtered = _execute_arxiv_search(current_query, max_count=3)

            if results_filtered:
                print(f"{C_GREEN} >> [SUCCESS] Attempt {i+1} ({attempt_name}) found {len(results_filtered)} papers. Stopping.{C_RESET}")
                break # Success, stop trying other queries
            else:
                print(f"{C_YELLOW} >> [WARN] Attempt {i+1} ({attempt_name}) returned 0 papers. Proceeding to next fallback.{C_RESET}")

        except Exception as e:
            print(f"{C_RED} >> [ERROR] Attempt {i+1} failed ({str(e)[:50]}...). Proceeding to next fallback.{C_RESET}")

    # --- 4. Final Processing and Reporting (Remains the same) ---
    processed_data = []

    if not results_filtered:
        print(f"{C_YELLOW} >> [WARN] Found 0 papers total after all fallbacks.{C_RESET}")

    print(f"{C_BLUE} >> [RESULTS] Final Papers and Summaries:{C_RESET}")

    for i, r in enumerate(results_filtered):
        summary_snippet = r.summary[:150].replace('\n', ' ')
        print(f"{C_GREEN}    > Paper {i+1} Title: {r.title[:80]}... (Year: {r.published.year}){C_RESET}")
        print(f"{C_GREEN}      Summary: '{summary_snippet}...'{C_RESET}")

        processed_data.append({
            "title": r.title,
            "summary": r.summary[:200],
            "pdf_url": r.pdf_url,
            "entry_id": r.entry_id
        })
        refs.append(f"🔗 ArXiv: {r.entry_id} (Year: {r.published.year})")

    multi["arxiv_search"] = processed_data
    print(f"\n{C_GREEN} >> [SUCCESS] Finished ArXiv search. **{len(processed_data)}** papers collected.{C_RESET}")

    print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")
    return state


def tool_mp(state: ResearchState) -> ResearchState:
    print(f"\n{C_MAGENTA}--------------------------------------------------{C_RESET}")
    print(f"{C_MAGENTA}[STEP 4.MP] MATERIALS PROJECT START{C_RESET} | Retrieving material properties (ID/Formula/Element search)...{C_RESET}")
    print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")

    extracted_elements = state.get("material_elements", [])
    api_search_term = state.get("api_search_term", "")
    multi = state.setdefault("multi_tool_data", {})
    refs = state.setdefault("references", [])

    if not MP_API_KEY:
        print(f"{C_RED} >> [FATAL] Missing MP API Key. Cannot proceed with Materials Project query.{C_RESET}")
        multi["materials_project"] = {"error": "API Key Missing"}
        print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")
        return state

    original_term = api_search_term
    if '\\text' in api_search_term or '$' in api_search_term or '_' in api_search_term:
        cleaned_term = original_term.replace('\\text{', '').replace('}', '')
        cleaned_term = cleaned_term.replace('$', '').replace('\\(', '').replace('\\)', '')
        cleaned_term = cleaned_term.replace('_', '')
        api_search_term = cleaned_term
        print(f"{C_YELLOW} >> [CLEAN] Cleaned LaTeX formula '{original_term}' to: **{api_search_term}**{C_RESET}")

    mp_id_pattern = re.compile(r'(mp-\d+)')
    mp_ids_to_query = mp_id_pattern.findall(api_search_term)

    query_params = {}
    query_description = ""
    max_results = 5

    if mp_ids_to_query:
        query_params["material_ids"] = mp_ids_to_query
        query_description = f"ID: {mp_ids_to_query}"
        max_results = len(mp_ids_to_query)

    elif extracted_elements and re.search(r'[A-Za-z]\d*([A-Za-z]\d*)+', api_search_term):
        query_params["formula"] = api_search_term
        query_description = f"Formula: {api_search_term}"
        max_results = 1

    elif extracted_elements:
        query_params["elements"] = extracted_elements
        query_description = f"Elements: {extracted_elements}"

    else:
        print(f"{C_YELLOW} >> [WARN] No valid Materials Project ID, Formula, or Elements found. Skipping MP API call.{C_RESET}")
        multi["materials_project"] = []
        print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")
        return state

    try:
        print(f"{C_CYAN} >> [INPUT] Querying by: **{query_description}** (Max results: {max_results}){C_RESET}")
        print(f"{C_BLUE} >> [ACTION] Calling MP API via MPRester...{C_RESET}")

        with MPRester(MP_API_KEY) as mpr:
            required_fields = ["material_id", "formula_pretty", "is_stable", "band_gap"]

            docs = mpr.materials.summary.search(
                **query_params,
                fields=required_fields
            )[:max_results]

        results = []
        stable_count = 0
        print(f"{C_CYAN} >> [INFO] Processing {len(docs)} documents returned from MP.{C_RESET}")

        for i, d in enumerate(docs):
            material_id = str(getattr(d, "material_id", "N/A"))
            formula = str(getattr(d, "formula_pretty", "N/A"))
            stability = getattr(d, "is_stable", False)
            band_gap = getattr(d, "band_gap", "N/A")

            try:
                formula = formula.encode('ascii', 'ignore').decode('utf-8')
            except Exception:
                pass

            if stability: stable_count += 1

            results.append({
                "id": material_id,
                "formula": formula,
                "stability": stability,
                "band_gap": band_gap
            })

            refs.append(f"⚛️ Materials Project ID: {material_id}")

            stability_str = f"{C_GREEN}Stable{C_RESET}" if stability else f"{C_YELLOW}Unstable{C_RESET}"
            print(f"    > Result {i+1}: ID **{material_id}** | Formula: {formula} | Status: {stability_str} | Band Gap: {band_gap}")

        multi["materials_project"] = results
        print(f"{C_GREEN} >> [SUCCESS] Retrieved **{len(results)}** materials. **{stable_count}** were stable.{C_RESET}")

    except Exception as e:
        print(f"{C_RED} >> [ERROR] MP API failed: {e}{C_RESET}")
        multi["materials_project"] = {"error": str(e)}

    print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")
    return state


################################################################################
# TOOL 3 (FINAL CORRECTED): Vector Search (Semantic Aggregation with Dual Filters)
################################################################################
def tool_vector(state: ResearchState) -> ResearchState:
    print(f"\n{C_MAGENTA}--------------------------------------------------{C_RESET}")
    print(f"{C_MAGENTA}[STEP 4.VECTOR] VECTOR SEARCH START{C_RESET} | Aggregating and filtering all parallel data...")
    print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")

    semantic_query = state.get("semantic_query", state.get("query", ""))
    literal_term = state.get("api_search_term", "").lower()
    multi = state.get("multi_tool_data", {})

    if not semantic_query:
        print(f"{C_RED} >> [FATAL] Semantic query is empty. Aborting vector search.{C_RESET}")
        state["filtered_context"] = "Error: No semantic query available for filtering."
        return state

    # 1. Chunk and Prepare Data for Embedding
    print(f"{C_CYAN} >> [ACTION] Granular chunking of {len(multi)} tool outputs...{C_RESET}")
    text_blocks = _granular_chunker(multi)

    if not text_blocks:
        print(f"{C_YELLOW} >> [WARN] No data available for embedding. Skipping RAG.{C_RESET}")
        state["filtered_context"] = "No external data was retrieved or processed."
        return state

    # 2. Generate Embeddings
    print(f"{C_ACTION} >> [ACTION] Generating embeddings for {len(text_blocks)} chunks...{C_RESET}")
    try:
        embeddings_response = client.embeddings.create(
            model=EMBED_MODEL,
            input=text_blocks
        )
        data_embeddings = np.array([item.embedding for item in embeddings_response.data]).astype("float32")
    except Exception as e:
        print(f"{C_RED} >> [ERROR] Embedding generation failed: {e}{C_RESET}")
        state["filtered_context"] = "Error: Embedding generation failed. Context unavailable."
        return state

    # 3. Build In-Memory FAISS Index
    index = faiss.IndexFlatL2(DIMENSION)
    index.add(data_embeddings)
    print(f"{C_GREEN} >> [SUCCESS] Temporary FAISS index built with {index.ntotal} vectors.{C_RESET}")

    # 4. Embed the Semantic Query & Perform Search
    try:
        query_embedding_response = client.embeddings.create(
            model=EMBED_MODEL,
            input=[semantic_query]
        )
        query_vector = np.array(query_embedding_response.data[0].embedding).astype("float32").reshape(1, -1)
    except Exception as e:
        print(f"{C_RED} >> [ERROR] Query embedding failed: {e}{C_RESET}")
        state["filtered_context"] = "Error: Query embedding failed. Context unavailable."
        return state

    K = min(10, index.ntotal)
    D, I = index.search(query_vector, K)
    initial_results = list(zip(I[0], D[0]))

    print(f"{C_CYAN} >> [INFO] Retrieving top {K} chunks semantically.{C_RESET}")

    # --- 5. CRITICAL: RERANKING AND DUAL FILTERING ---
    filtered_context_lines = []

    DISTANCE_THRESHOLD = 1.2

    print(f"{C_ACTION} >> [RAG FILTER] Applying Keyword Gate and Distance Threshold ({DISTANCE_THRESHOLD}).{C_RESET}")

    for rank, (chunk_index, distance) in enumerate(initial_results):
        if chunk_index < 0: continue

        # 1. Apply HARD DISTANCE CUTOFF (Filter 1)
        if distance > DISTANCE_THRESHOLD:
            print(f"{C_YELLOW}    > [FILTER 1] Discarding due to high distance ({distance:.4f} > {DISTANCE_THRESHOLD}). Stopping retrieval.{C_RESET}")
            break

        chunk_text = text_blocks[chunk_index]

        # 2. Apply KEYWORD GATE (Filter 2 - Targets Academic Noise)
        is_academic_noise = "full text expander" in chunk_text.lower() or "arxiv paper" in chunk_text.lower()
        contains_literal = literal_term in chunk_text.lower()

        print(f"{C_BLUE}    > Rank {rank+1} (Dist: {distance:.4f}) | Academic: {is_academic_noise} | Contains Literal: {contains_literal}{C_RESET}")

        if is_academic_noise and not contains_literal:
            print(f"{C_YELLOW}    > [FILTER 2] Discarding general academic context (lacks specific term).{C_RESET}")
            continue

        try:
            cleansed_chunk = chunk_text.encode('ascii', 'ignore').decode('utf-8')
        except Exception:
            cleansed_chunk = chunk_text

        filtered_context_lines.append(cleansed_chunk)

        if len(filtered_context_lines) >= 5:
             break

    # --- END OF DUAL FILTERING ---

    if not filtered_context_lines:
        final_context = "No sufficiently relevant or specific context was found to address the user's query."
        print(f"{C_RED} >> [FAILURE] Filtering resulted in empty context. Forcing negative answer for Synthesis.{C_RESET}")
    else:
        final_context = "\n---\n".join(filtered_context_lines)

    state["filtered_context"] = final_context

    print(f"{C_GREEN} >> [SUCCESS] Aggregation and filtering complete. Context size: {len(final_context)} chars (Filtered {len(initial_results) - len(filtered_context_lines)} chunks from initial top {K}).{C_RESET}")
    print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")
    return state


################################################################################
# TOOL X: Python Compute (Placeholder)
################################################################################
def tool_python(state: ResearchState) -> ResearchState:
    print(f"{C_MAGENTA}[STEP 4.Python] tool_python{C_RESET} | Executing (Placeholder for code execution)...")
    state.setdefault("multi_tool_data", {})["python_compute"] = {"result": "Computational analysis is not currently implemented."}
    print(f"{C_YELLOW} >> [WARN] Python compute is a placeholder and was skipped.{C_RESET}")
    return state

################################################################################
# TOOL 5: OpenAlex Search
################################################################################
def tool_openalex(state: ResearchState) -> ResearchState:
    print(f"\n{C_MAGENTA}--------------------------------------------------{C_RESET}")
    print(f"{C_MAGENTA}[STEP 4.OpenAlex] OPENALEX SEARCH START{C_RESET} | Retrieving open-access data...")
    print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")
    query = state.get("api_search_term", state.get("clean_query", ""))
    multi = state.setdefault("multi_tool_data", {})

    if not query:
        print(f"{C_RED} >> [ERROR] Tool skipping execution due to empty query.{C_RESET}")
        multi["openalex_search"] = {"error": "Empty search query provided."}
        return state

    try:
        url = f"https://api.openalex.org/works?filter=title.search:{query}&per-page=3"
        r = requests.get(url)
        r.raise_for_status()

        data = r.json().get("results", [])
        multi["openalex_search"] = data
        print(f"{C_GREEN} >> [SUCCESS] Found {len(data)} works.{C_RESET}")
    except Exception as e:
        print(f"{C_RED} >> [ERROR] OpenAlex failed: {e}{C_RESET}")
        multi["openalex_search"] = {"error": str(e)}
    print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")
    return state

################################################################################
# TOOL 6: PubMed (NCBI) Search
################################################################################
# --- Helper Function for Boolean Query Construction (MUST BE DEFINED GLOBALLY/IMPORTED) ---
# NOTE: This function should be placed before tool_pubmed or imported.

def _construct_strict_boolean_query(semantic_query: str) -> str | None:
    """
    Constructs a strict Boolean query if high specificity is detected (Technology, App, Method),
    otherwise returns None.
    """
    query_lower = semantic_query.lower()

    # Define categories of keywords for specificity check
    tech_terms = ["aptamer", "biosensor", "detection", "sensor"]
    app_terms = ["cancer", "tumor", "diagnosis", "disease"]
    method_terms = ["machine learning", "ai", "artificial intelligence", "ml"]

    # Check for presence in all three categories
    has_tech = any(term in query_lower for term in tech_terms)
    has_app = any(term in query_lower for term in app_terms)
    has_method = any(term in query_lower for term in method_terms)

    if has_tech and has_app and has_method:
        print(f"{C_BLUE} >> [INFO] Strict Boolean search criteria met.{C_RESET}")

        # Core Tech (Must be present)
        core_tech = '("aptamer" OR "biosensor" OR "sensor")'

        # Target/Application (Must be present)
        target_app = '("cancer" OR "tumor" OR "diagnosis" OR "disease")'

        # Methodology (Must be present)
        methodology = '("machine learning" OR "AI" OR "ML" OR "artificial intelligence")'

        # Final query enforcing AND for all concepts and OR for synonyms
        strict_query = f'{core_tech} AND {target_app} AND {methodology}'
        return strict_query

    return None # Return None if not highly specific

# --- TOOL 6: PubMed (NCBI) Search ---

def tool_pubmed(state: ResearchState) -> ResearchState:
    print(f"\n{C_MAGENTA}--------------------------------------------------{C_RESET}")
    print(f"{C_MAGENTA}[STEP 4.PubMed] PUBMED SEARCH START{C_RESET} | Executing search for IDs with fallback...")
    print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")

    multi = state.setdefault("multi_tool_data", {})
    Entrez.email = "research.agent@example.com"

    complex_query = state.get("semantic_query", "")
    simple_query = state.get("api_search_term", state.get("clean_query", ""))

    # Define the ordered list of queries to attempt
    query_attempts = []

    # 1. Highest Precision: Strict Boolean Query (if applicable)
    strict_boolean_query = _construct_strict_boolean_query(complex_query)
    if strict_boolean_query:
        query_attempts.append(strict_boolean_query)

    # 2. Medium Precision: Original Semantic Query
    if complex_query:
        query_attempts.append(complex_query)

    # 3. Lowest Precision: Simple API Search Term
    if simple_query and simple_query not in query_attempts:
        query_attempts.append(simple_query)

    if not query_attempts:
        print(f"{C_RED} >> [ERROR] No valid search terms available. Final failure.{C_RESET}")
        multi["pubmed_search"] = {"error": "No valid search terms available."}
        print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")
        return state

    final_search_term = ""
    ids = []

    # --- Execute Phased Search Attempts ---
    for i, current_query in enumerate(query_attempts):
        attempt_name = "STRICT BOOLEAN" if i == 0 and strict_boolean_query else ("COMPLEX SEMANTIC" if i == 0 or i == 1 and current_query == complex_query else "SIMPLE LITERAL")

        try:
            final_search_term = current_query
            print(f"{C_BLUE} >> [ACTION] Attempt {i+1} ({attempt_name}): {final_search_term[:80]}...{C_RESET}")

            handle = Entrez.esearch(db="pubmed", term=final_search_term, retmax=5, sort="relevance")
            ids = Entrez.read(handle).get("IdList", [])

            if ids:
                print(f"{C_GREEN} >> [SUCCESS] Found IDs: {ids}{C_RESET}")
                break # Success, stop trying other queries
            else:
                print(f"{C_YELLOW} >> [WARN] Attempt {i+1} returned 0 IDs. Proceeding to next fallback.{C_RESET}")

        except Exception as e:
            print(f"{C_YELLOW} >> [WARN] Attempt {i+1} failed ({str(e)[:50]}...). Proceeding to next fallback.{C_RESET}")

    # --- Final Result Reporting ---
    multi["pubmed_search"] = [{"id": i, "source": "PubMed"} for i in ids]

    if not ids:
        print(f"{C_RED} >> [ERROR] All search attempts failed. 0 IDs found.{C_RESET}")

    print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")
    return state


################################################################################
# TOOL (Helper): PUBMED METADATA LOOKUP
################################################################################
def tool_pubmed_metadata_lookup(state: Dict[str, Any]) -> Dict[str, Any]:
    print(f"\n{C_MAGENTA}--------------------------------------------------{C_RESET}")
    print(f"{C_MAGENTA}[STEP 4.Meta] PUBMED METADATA LOOKUP START{C_RESET} | Fetching full citation details...")
    print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")

    Entrez.email = "research.agent@example.com"
    pubmed_results = state.get("multi_tool_data", {}).get("pubmed_search", [])
    pmids: List[str] = [item['id'] for item in pubmed_results if isinstance(item, dict) and 'id' in item and item['id'].isdigit()]

    if not pmids:
        print(f"{C_YELLOW} >> [WARN] No valid PMIDs found for lookup. Skipping.{C_RESET}")
        return state

    print(f"{C_CYAN} >> [ACTION] Found {len(pmids)} PMIDs: {pmids}. Executing Entrez.efetch...{C_RESET}")

    citation_details = []

    try:
        handle = Entrez.efetch(
            db="pubmed",
            id=pmids,
            rettype="medline",
            retmode="xml"
        )
        records = Entrez.read(handle)
        handle.close()

        for record in records.get('PubmedArticle', []):
            medline_citation = record.get('MedlineCitation', {})
            article = medline_citation.get('Article', {})
            pmid = str(medline_citation.get('PMID', 'N/A'))

            title = str(article.get('ArticleTitle', 'No Title Available')).strip()
            journal_title = str(article.get('Journal', {}).get('Title', 'No Journal')).strip()
            year = 'N.D.'

            pub_date_info = article.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
            year_candidate = pub_date_info.get('Year', pub_date_info.get('MedlineDate'))

            if not year_candidate:
                article_date = article.get('ArticleDate', {})
                if isinstance(article_date, list) and article_date:
                    year_candidate = article_date[0].get('Year')

            if year_candidate:
                try:
                    if isinstance(year_candidate, str):
                        match = re.search(r'\b(20\d{2})\b', year_candidate)
                        year = match.group(1) if match else year_candidate
                    else:
                        year = str(year_candidate)
                except Exception:
                    pass

            full_citation = f"📄 Journal Article: {title}. *{journal_title}*. Published: {year}. (PMID: {pmid})"
            citation_details.append(full_citation)
            print(f"{C_GREEN}    > Formatted Citation: {full_citation[:100]}...{C_RESET}")


        state.setdefault("references", []).extend(citation_details)
        print(f"{C_GREEN} >> [SUCCESS] Retrieved and formatted {len(citation_details)} full citations.{C_RESET}")

    except Exception as e:
        print(f"{C_RED} >> [ERROR] PubMed Metadata Lookup failed: {e}{C_RESET}")

    print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")
    return state


################################################################################
# HELPER: _split_text_into_chunks
################################################################################
def _split_text_into_chunks(text: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> List[str]:
    """Simple text splitter to break large documents into overlapping chunks."""
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
        i += chunk_size - chunk_overlap
    return chunks


################################################################################
# TOOL 7: tool_full_text_expander
################################################################################
def tool_full_text_expander(state: ResearchState) -> ResearchState:
    print(f"\n{C_MAGENTA}--------------------------------------------------{C_RESET}")
    print(f"{C_MAGENTA}[STEP 4.EXPANDER] FULL-TEXT EXPANDER START{C_RESET} | Retrieving full papers...")
    print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")

    multi = state.setdefault("multi_tool_data", {})
    arxiv_results = multi.get("arxiv_search", [])

    if not arxiv_results or isinstance(arxiv_results, dict) and "error" in arxiv_results:
        print(f"{C_YELLOW} >> [WARN] No valid ArXiv results to expand. Skipping.{C_RESET}")
        multi["full_text_expander"] = {"status": "Skipped"}
        return state

    full_text_chunks = []
    papers_to_process = arxiv_results[:2]

    for i, paper in enumerate(papers_to_process):
        pdf_url = paper.get("pdf_url")
        title = paper.get("title", f"Paper {i+1}")

        if not pdf_url:
            print(f"{C_YELLOW} >> [WARN] Skipping '{title}'. No PDF URL found.{C_RESET}")
            continue

        print(f"{C_CYAN} >> [ACTION] Attempting download for: '{title[:50]}...'{C_RESET}")

        try:
            time.sleep(1)
            response = requests.get(pdf_url, stream=True, timeout=10)
            response.raise_for_status()

            pdf_file_in_memory = io.BytesIO(response.content)
            reader = PdfReader(pdf_file_in_memory)

            raw_text = ""
            for page in reader.pages:
                raw_text += page.extract_text() or ""

            chunks = _split_text_into_chunks(raw_text, chunk_size=3000, chunk_overlap=300)
            print(f"{C_GREEN} >> [SUCCESS] Extracted {len(raw_text)//1000}k chars, split into **{len(chunks)}** chunks.{C_RESET}")

            for chunk_num, chunk in enumerate(chunks):
                full_text_chunks.append(
                    f"Source: Full Text - {title}. Chunk {chunk_num+1}/{len(chunks)}. Content: {chunk.strip()}"
                )

        except Exception as e:
            print(f"{C_RED} >> [ERROR] Failed to download/parse '{title}': {e}{C_RESET}")
            continue

    if full_text_chunks:
        multi["full_text_expander"] = full_text_chunks
        print(f"{C_GREEN} >> [SUCCESS] Full text added to multi_tool_data for vector search.{C_RESET}")
    else:
        multi["full_text_expander"] = {"status": "No full text retrieved"}

    print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")
    return state


################################################################################
# TOOL 8: Patent Search (Simulated)
################################################################################
def tool_patent_search(state: ResearchState) -> ResearchState:
    print(f"{C_MAGENTA}[STEP 4.Patent] tool_patent_search{C_RESET} | Executing...")
    query = state.get("api_search_term", state.get("clean_query", ""))

    if not query:
        query = "placeholder_query"

    simulated_patent = {
        "title": f"Method for synthesizing advanced {query} cathode material",
        "inventor": "J. Doe",
        "date": "2024-01-15"
    }
    state.setdefault("multi_tool_data", {})["patent_search"] = [simulated_patent]
    print(f"{C_GREEN} >> [SUCCESS] Patent search checked (Simulated). Query: {query}{C_RESET}")
    return state


################################################################################
# HELPER: _call_ddg_search
################################################################################
def _call_ddg_search(query: str, limit: int = 5) -> List[str]:
    """Uses DDGS to perform a general web search, with improved error reporting."""
    results = []
    print(f"{C_CYAN} >> [ACTION] Attempting DDG search for: '{query}' (Limit: {limit}){C_RESET}")

    try:
        print(f"{C_BLUE}    > [DEBUG A] Starting DDGS object initialization...{C_RESET}")
        ddgs = DDGS()
        print(f"{C_BLUE}    > [DEBUG B] DDGS object successfully initialized. Executing text search...{C_RESET}")

        search_results = ddgs.text(query=query, max_results=limit)
        search_list = list(search_results)

        print(f"{C_BLUE}    > [DEBUG C] DDGS search execution complete. Results found: {len(search_list)}{C_RESET}")

        if not search_list:
            print(f"{C_YELLOW} >> [DDG] Search returned an empty list or generator.{C_RESET}")
            return []

        for i, r in enumerate(search_list):
            title = r.get('title', 'No Title')
            snippet = r.get('body', 'No snippet available')
            url = r.get('href', 'N/A')

            block = f"Source: Google Search Result {i+1}. Title: {title}. Snippet: {snippet}. URL: {url}"
            results.append(block)

        print(f"{C_GREEN} >> [DDG] Successfully retrieved {len(results)} raw snippets.{C_RESET}")
        return results

    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        print(f"{C_RED} >> [FATAL ERROR] DDG Search failed with exception: {error_type} - {error_message}{C_RESET}")
        traceback.print_exc()
        return []

################################################################################
# TOOL 9: tool_google_search (Web Search Safety Net - Enhanced Debugging)
################################################################################
def tool_google_search(state: ResearchState) -> ResearchState:
    print(f"\n{C_MAGENTA}--------------------------------------------------{C_RESET}")
    print(f"{C_MAGENTA}[STEP 4.Web] GOOGLE SEARCH START{C_RESET} | Retrieving general context...")
    print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")

    focused_query = state.get("semantic_query", state.get("api_search_term", ""))
    multi = state.setdefault("multi_tool_data", {})
    refs = state.setdefault("references", [])

    if not focused_query:
        print(f"{C_RED} >> [ERROR] Tool skipping execution due to empty query.{C_RESET}")
        multi["google_search"] = {"error": "Empty search query provided."}
        return state

    print(f"{C_BLUE}    > [DEBUG D] Calling _call_ddg_search helper...{C_RESET}")
    rag_blocks = _call_ddg_search(query=focused_query, limit=5)
    print(f"{C_BLUE}    > [DEBUG E] _call_ddg_search returned {len(rag_blocks)} blocks.{C_RESET}")

    if not rag_blocks:
        print(f"{C_YELLOW} >> [WARN] Google Search (via DDG) returned no results or failed execution. Proceeding with workflow...{C_RESET}")
        multi["google_search"] = {"status": "No results or Failed to run"}
        print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")
        return state

    multi["google_search"] = rag_blocks
    added_ref_count = 0

    print(f"{C_GREEN} >> [SUCCESS] Retrieved {len(rag_blocks)} web snippets. {added_ref_count} references added.{C_RESET}")
    print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")
    return state


################################################################################
# HELPER: _cleanse_text_data (Ensures safety for prompt/output)
################################################################################
def _cleanse_text_data(text: str) -> str:
    """Removes problematic Unicode surrogate characters (U+D800 to U+DFFF)."""
    if not isinstance(text, str):
        return ""

    surrogate_pattern = re.compile(r'[\ud800-\udfff]')
    safe_text = surrogate_pattern.sub('', text)

    try:
        return safe_text.encode('utf-8', 'ignore').decode('utf-8').strip()
    except Exception:
        return safe_text.strip()


################################################################################
# TOOL 10 (FINAL CORRECTED): Synthesis (Final Report Generation with Guardrail and Reference Filter)
################################################################################
def tool_synthesis(state: ResearchState) -> ResearchState:
    print(f"\n{C_MAGENTA}--------------------------------------------------{C_RESET}")
    print(f"{C_MAGENTA}[STEP 5] SYNTHESIS START{C_RESET} | Generating final report from aggregated context...")
    print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")

    context = state.get("filtered_context", "")
    query = state.get('query', '')
    references = state.get('references', [])
    api_search_term = state.get('api_search_term', query)

    if not context or "No sufficiently relevant or specific context was found" in context:
        print(f"{C_RED} >> [FATAL] No filtered context was available or context explicitly failed. Generating fallback answer.{C_RESET}")

        if "No sufficiently relevant or specific context was found" in context:
             fallback_answer = f"**Research Summary:**\n\nThe research process was unable to locate specific references or data for the discovery of **'{api_search_term}'** in academic databases or general web search results. This suggests the requested term may not correspond to a known scientific finding."
        else:
             fallback_answer = f"The research process failed to retrieve specific data. Based on your general knowledge, answer the following query: {query}"

        final_prompt = fallback_answer

    else:
        # --- CRITICAL FIX 1: CLEANSING AND FILTERING REFERENCES LIST BEFORE PROMPT ---
        cleansed_references = []
        # Keywords highly relevant to the biosensing query
        REQUIRED_KEYWORDS = ["biosens", "immunosens", "detection", "analy", "nanozym", "diagno", "electrochemical"]

        for ref in references:
            # 1. Cleanse the reference text
            try:
                cleansed_ref = _cleanse_text_data(str(ref))
            except Exception:
                cleansed_ref = "[Reference Cleansing Error]"

            ref_lower = cleansed_ref.lower()

            # --- FINAL REFERENCE FILTERING LOGIC ---
            keep_ref = True

            is_academic_noise_source = "arxiv:" in ref_lower or "openalex:" in ref_lower

            # FIX: Only filter ArXiv/OpenAlex references if they lack the strong biosensing keywords.
            # PubMed (Journal Article) is implicitly trusted to pass.
            if is_academic_noise_source and "journal article:" not in ref_lower:
                if not any(kw in ref_lower for kw in REQUIRED_KEYWORDS):
                    keep_ref = False

            if "[Reference Cleansing Error]" in cleansed_ref:
                keep_ref = False

            if keep_ref:
                cleansed_references.append(cleansed_ref)
            else:
                if is_academic_noise_source:
                     print(f"{C_YELLOW} >> [REF FILTER] Dropping noisy academic reference lacking keywords: {cleansed_ref[:50]}...{C_RESET}")


        reference_list = "\n".join([f"* {r}" for r in cleansed_references])
        # ---------------------------------------------------------------------------------

        print(f"{C_CYAN} >> [RAG] Using context lines: **{len(context.splitlines())}**{C_RESET}")

        if not reference_list:
            reference_section = "## 📚 Citations & References\nNo specific, supporting citations were found for the final synthesized content."
        else:
            reference_section = f"""
            ## 📚 Citations & References
            **You must include every single item from the provided list exactly as it is presented.**

            **NEW INSTRUCTION: VISUAL STYLING**
            Prepend each reference type with a relevant emoji, for example:
            * **PubMed/Journal Articles** should be preceded by a **📄** (Document) or **🔬** (Microscope) emoji.
            * **ArXiv/OpenAlex/Web Sources** should be preceded by a **🔗** (Link) or **🌐** (Globe) emoji.
            * **Materials Project IDs** should be preceded by a **⚛️** (Atom) emoji.

            {reference_list}
            """


        final_prompt = f"""
        You are an expert research analyst.
        Your task is to analyze the user's request and synthesize a comprehensive, cohesive, and easy-to-read report based **ONLY** on the
        context provided below.

        **CRITICAL ANTI-HALLUCINATION GUARDRAIL:**
        The user's specific search term was **'{api_search_term}'**.
        If the AGGREGATED CONTEXT does not explicitly mention this specific term, or if the context only contains fragments of general information:
        1. You **MUST** explicitly state in the report that the specific reference/discovery for **'{api_search_term}'** could not be located in the sources.
        2. You may then, optionally, provide general, related background on the broader topic (e.g., 'gravity waves') found in the context, but must clarify that this is **not** a reference for the specific term requested.

        1. **Do not** mention the context, sources, or documents directly in the answer body (e.g., do not say "According to Source: ArXiv Paper...").
        2. **Organize** the information clearly using Markdown headings (##) and bullet points.
        3. The answer **must be comprehensive** and directly address the user's query.
        4. **CRITICAL:** Include the provided **Citations and References** section at the very end of your report exactly as it is given below.

        ---
        USER QUERY: {query}
        ---
        AGGREGATED, FILTERED CONTEXT (This is your ONLY source of truth):
        {context}
        ---
        {reference_section}
        ---
        """
    try:
        print(f"{C_YELLOW} >> [ACTION] Sending prompt to LLM for synthesis...{C_RESET}")

        if not context or "No sufficiently relevant or specific context was found" in context:
            final_answer = final_prompt
        else:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.2,
                max_tokens=4096
            )
            final_answer = resp.choices[0].message.content.strip()

        cleansed_answer = _cleanse_text_data(final_answer)
        state["answer"] = cleansed_answer

        print(f"{C_GREEN} >> [SUCCESS] Final report synthesized. Length: {len(cleansed_answer)} chars.{C_RESET}")

    except Exception as e:
        clean_error_message = _cleanse_text_data(str(e))
        state["answer"] = f"An internal error occurred during the final synthesis step: {clean_error_message}"
        print(f"{C_RED} >> [ERROR] Synthesis failed: {e}{C_RESET}")

    print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")
    print(state.get('answer', ''))
    return state

# ==================================================================================================
# SECTION 7: ORCHESTRATION NODE (CRITICAL)
# ==================================================================================================
def run_secondary_tools(state: ResearchState) -> ResearchState:
    print(f"\n{C_MAGENTA}--------------------------------------------------{C_RESET}")
    print(f"{C_MAGENTA}[STEP 4.Secondary] RUN SECONDARY TOOLS START{C_RESET} | Executing cross-reference tools...")
    print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")

    secondary_tools = state.get("secondary_tools", [])

    tools_to_run = [
        tool_name for tool_name in secondary_tools
        if tool_name not in [state.get("intent"), "vector_search"]
    ]

    print(f"{C_CYAN} >> [ACTION] Tools to execute: {tools_to_run}{C_RESET}")
    # === INJECT THE METADATA LOOKUP TOOL ===
    if "pubmed_search" in tools_to_run and "pubmed_metadata_lookup" not in tools_to_run:
        try:
            pubmed_index = tools_to_run.index("pubmed_search")
            tools_to_run.insert(pubmed_index + 1, "pubmed_metadata_lookup")
            print(f"{C_CYAN} >> [INFO] Injected 'pubmed_metadata_lookup' after 'pubmed_search' for proper sequencing.{C_RESET}")
        except ValueError:
            pass

    node_function_map = {
        "arxiv_search": tool_arxiv,
        "materials_project": tool_mp,
        "google_search": tool_google_search,
        "paper_retrieve": tool_full_text_expander,
        "python_compute": tool_python,
        "openalex_search": tool_openalex,
        "pubmed_search": tool_pubmed,
        "pubmed_metadata_lookup": tool_pubmed_metadata_lookup,
        "patent_search": tool_patent_search,
    }

    temp_state = state.copy()

    for tool_key in tools_to_run:
        tool_func = node_function_map.get(tool_key)
        if tool_func:
            print(f"{C_BLUE}    > Running Secondary Tool: {tool_key}{C_RESET}")
            try:
                temp_state = tool_func(temp_state)
            except Exception as e:
                print(f"{C_RED}    > [ERROR] Secondary tool {tool_key} failed: {e}{C_RESET}")
        else:
            print(f"{C_YELLOW}    > [WARN] Function for tool {tool_key} not found or skipped.{C_RESET}")

    print(f"{C_GREEN} >> [SUCCESS] Secondary tool execution complete.{C_RESET}")
    print(f"{C_MAGENTA}--------------------------------------------------{C_RESET}")
    return temp_state

# ==================================================================================================
# SECTION 8: GRAPH ASSEMBLY (FINAL CORRECTED)
# ==================================================================================================
print(f"{C_GREEN}[INIT] Assembling Research Agent Workflow Graph...{C_RESET}")

workflow = StateGraph(ResearchState)

# 1. Register all necessary nodes
nodes = [
    clean_query, detect_intent, route_to_tool,
    tool_arxiv, tool_mp, tool_vector, tool_python,
    tool_openalex, tool_pubmed, tool_google_search, tool_full_text_expander, tool_patent_search,
    tool_pubmed_metadata_lookup,run_secondary_tools, tool_synthesis
]
for n in nodes:
    workflow.add_node(n.__name__, n)
    print(f"{C_CYAN} >> [GRAPH INIT] Node added: {n.__name__}{C_RESET}")

# 2. Define Entry Logic
workflow.set_entry_point("clean_query")
workflow.add_edge("clean_query", "detect_intent")
workflow.add_edge("detect_intent", "route_to_tool")

# 3. Define Conditional Routing (Primary Tool Selection)
tool_names_for_routing = [
    "tool_arxiv", "tool_mp", "tool_python",
    "tool_openalex", "tool_pubmed", "tool_patent_search",
    "tool_google_search"
]
router_map = {name: name for name in tool_names_for_routing}

def router_logic(state):
    return state.get("tool", "tool_arxiv")

workflow.add_conditional_edges("route_to_tool", router_logic, router_map)

# 4. Define Parallel/Sequence Flow and Convergence (The Diamond Join)
for name in tool_names_for_routing:
    workflow.add_edge(name, "run_secondary_tools")

# 5. Define Analysis Flow (RAG Pipeline)
workflow.add_edge("run_secondary_tools", "tool_vector")
workflow.add_edge("tool_vector", "tool_synthesis")
workflow.add_edge("tool_synthesis", END)

# 6. Compile
research_agent_app = workflow.compile()
print(f"{C_GREEN}[INIT] Research Agent Graph compiled successfully.{C_RESET}")


# ==================================================================================================
# SECTION 9: VISUALIZATION UTILITY (FONT SIZE BALANCED, CURVED EDGES)
# ==================================================================================================
def visualize_graph(app, filename="research_agent_v7") -> bytes:
    """
    Generates a professionally styled, colorful PNG visualization of the workflow,
    with increased font size for better readability and curved edges.
    """
    try:
        g = app.get_graph(xray=True)
        dot = Digraph(comment="Research Agent v7", format="png")
        dot.attr(rankdir="TB")
        dot.attr(splines="curved")
        dot.attr(nodesep="0.4")
        dot.attr(ranksep="0.8")
        dot.attr(fontname="Arial", fontsize="12")

        for n in g.nodes:
            node_id = n.id if hasattr(n, "id") else str(n)
            attrs = {
                "style": "filled,rounded",
                "fontname": "Arial",
                "fontsize": "12",
                "penwidth": "1.5"
            }

            if node_id == "__start__":
                attrs.update({"shape": "circle", "fillcolor": "#4CAF50", "fontcolor": "white", "label": "Start", "color": "#2E7D32"})
            elif node_id == "__end__":
                attrs.update({"shape": "doublecircle", "fillcolor": "#F44336", "fontcolor": "white", "label": "End", "color": "#C62828"})
            elif node_id in ["detect_intent", "route_to_tool", "clean_query"]:
                attrs.update({"shape": "diamond", "fillcolor": "#FFF59D", "color": "#FBC02D", "label": node_id})
            elif node_id.startswith("tool_"):
                attrs.update({"shape": "cylinder", "fillcolor": "#81D4FA", "color": "#0277BD", "label": node_id})
            elif node_id in ["analyze_raw_data", "final_answer"]:
                attrs.update({"shape": "note", "fillcolor": "#E1BEE7", "color": "#7B1FA2", "label": node_id})
            elif node_id in ["run_secondary_tools", "resolve_material_elements"]:
                attrs.update({"shape": "hexagon", "fillcolor": "#F5F5F5", "color": "#9E9E9E", "label": node_id})
            else:
                attrs.update({"shape": "box", "fillcolor": "white", "label": node_id})

            dot.node(node_id, **attrs)

        for e in g.edges:
            edge_attrs = {"color": "#424242", "penwidth": "1.0", "arrowsize": "0.8"}
            if e.data and "condition" in str(e.data):
                edge_attrs.update({"style": "dashed", "color": "#F57F17"})
            dot.edge(e.source, e.target, **edge_attrs)

        print(f"{C_CYAN}[VISUAL] Rendering colorful graph to {filename}...{C_RESET}")
        return dot.pipe(format="png")

    except Exception as e:
        print(f"{C_RED}[VISUAL ERROR] Graphviz visualization failed: {e}{C_RESET}")
        return b""

# ==================================================================================================
# SECTION 10: DEBUG ENTRY POINT
# ==================================================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print(f"{C_MAGENTA}      RESEARCH AGENT v7.0: FULL DEBUG MODE{C_RESET}")
    print("="*60)

    test_query = "Give me the properties of the material Graphene Oxide and list three recent journal articles (last 2 years) on its use in biosensing."
    print(f"\n{C_CYAN}[INVOKE] Starting agent with: '{test_query}'{C_RESET}")

    try:
        final_state = research_agent_app.invoke(
            {"query": test_query},
            config={"recursion_limit": 50}
        )

        print("\n" + "="*60)
        print(f"{C_MAGENTA}          FINAL REPORT SUMMARY{C_RESET}")
        print("="*60)
        print(f"Original Query: {final_state['query']}")
        print(f"Primary Intent: {final_state.get('intent')}")
        print(f"Secondary Tools: {final_state.get('secondary_tools')}")
        print(f"Data Sources:   {list(final_state.get('multi_tool_data', {}).keys())}")
        print("-" * 30)
        print(final_state.get('answer'))
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print(f"{C_RED}AGENT EXECUTION FAILED!{C_RESET}")
        print("="*60)
        traceback.print_exc()