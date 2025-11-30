# 🧪 Declarative Research Agent — Full Stack System
<img width="1641" height="857" alt="Screenshot 2025-11-30 at 23 03 22" src="https://github.com/user-attachments/assets/e9d039db-42a8-4553-9ba9-5fdc12479040" />



***Autonomous Scientific Research Pipeline with FastAPI, LangGraph, Streamlit, and SQLite Logging***

Version 3.3 — **NASA-Style Architecture**

---

## 🚀 Overview

The Declarative Research Agent is an end-to-end scientific research assistant that:
- Accepts natural language queries
- Performs intent detection
- Searches external data sources (ArXiv, PubMed, Google Search, OpenAlex)
- Downloads PDFs
- Runs vector-based semantic filtering
- Synthesizes a final scientific report with citations
- Stores full chat logs and tool outputs in SQLite
- Visualizes the full LangGraph workflow as a PNG

### The system is composed of three major modules:
1. LangGraph Workflow (research_agent_core.py / graph.py)
2. FastAPI Backend (backend.py)
3. Streamlit Frontend (main_ui.py)

---

## 📁 Project Structure
```Bash
research-agent/
│
├── backend/
│   ├── backend.py               # FastAPI server
│   ├── graph.py                 # Compiled LangGraph workflow + visualization
│   ├── chat_history.db          # SQLite DB (auto-generated)
│
├── core/
│   └── research_agent_core.py   # Merged LangGraph declarative logic
│
├── frontend/
│   └── main_ui.py               # Streamlit GUI
│
└── README.md                    # You're reading this!

```

---

## 🔧 1. Installation
### ✔ Clone the Repository
```Bash
git clone https://github.com/yourname/ResearchLLM_Single_Agent.git
cd ResearchLLM_Single_Agent
```
---
## 🔑 2. Environment Setup
### ✔ Create Python Environment
```Bash
python3 -m venv venv
source venv/bin/activate
```
---
### ✔ Install Dependencies
```Bash
fastapi
uvicorn
openai>=1.0.0
langgraph
langchain-community
streamlit
requests
pydantic
python-dotenv
arxiv
mp-api
faiss-cpu
numpy
sqlalchemy
databases
graphviz
Bio
langchain-core
pypdf
ddgs
```
---
## 🔐 3. Environment Variables

Create a .env file in the project root:
```Bash
MP_API_KEY=.....
GEMINI_API_KEY=......
GPT_API_KEY=......
GPT_5_API_KEY=....
```
---
## ▶️ 4. Running the Project
### Start Backend (FastAPI)
```Bash
uvicorn backend:app --host 0.0.0.0 --port 8000
or
uvicorn backend:app --reload
```
---

### Start Frontend (Streamlit)
```Bash
streamlit run ui_main.py
```
---
## 🧠 5. System Architecture
### 🎯 High-Level Workflow

```mermaid
graph TD
    %% ---------------- Initialization and Planning (Remains the same) ----------------
    subgraph Initialization
        A[User Query] --> B{STEP 1: clean_query}
        B --> C[State: Semantic Query]
        B --> D[State: API Search Term]
    end

    subgraph Planning
        C --> E{STEP 2: detect_intent}
        D --> E
        E --> F[State: Plan and Tool List]
    end

    %% ---------------- Data Retrieval Orchestration (CORRECTED SECTION) ----------------
    subgraph DataRetrieval
        %% Primary routing: The choice based on intent
        F --> G{STEP 3: route_to_tool}
        
        G --> G_P_choice{Primary Tool Choice};
        
        G_P_choice --> G_P1[tool_pubmed/
										        openalex/
										        arxiv...];
        G_P_choice --> G_P2[tool_google_search];

        %% The chosen primary tool runs first
        G_P1 & G_P2 --> G_S{Secondary Tool Executor};
        
        %% Secondary Tool Executor (G_S) launches all other tools in parallel
        G_S --> H[tool_pubmed/other tools]
        G_S --> I[tool_arxiv]
        G_S --> J[tool_openalex]
        G_S --> K[tool_google_search]

        %% Aggregation of Metadata
        L[Metadata: Citations 
		        and Abstracts]
        H --> L
        I --> L
        J --> L
        K --> L

        %% Full Text Retrieval
        L --> M{tool_paper_retrieve: 
				        Full PDF Download}
        
        %% Data feeds into the Secondary Executor to signal collection completion
        M --> G_S
        L --> G_S 
    end

    %% ---------------- Filtration and Synthesis (Remains the same) ----------------
    G_S --> N{STEP 4 VECTOR: 
				    vector_search_filter}
    
    subgraph Filtration
        L --> N
        M --> N

        N --> O[Vector Index Chunks 
				        Stored]
        N --> P{Semantic Retriever}

        P --> Q{Filter 1: Keyword Gate}
        Q --> R{Filter 2: Distance 
				        Threshold <= 1.2}

        R -->|Pass| S[Filtered Context]
        R -->|Fail| T[Discarded Noise]
    end

    subgraph Synthesis
        S --> U{STEP 5: tool_synthesis LLM}
        U --> V[Final Report 
				        and Citations]
        U --> W[Negative Report: 
				        No Evidence]
    end

    %% ----------- Styles -----------
    style B fill:#e6ffe6,stroke:#00aaff,stroke-width:2px
    style E fill:#fff2cc,stroke:#ffaa00
    style G fill:#ffcccc,stroke:#ff0000
    style N fill:#e6ccff,stroke:#8000ff,stroke-width:3px
    style S fill:#ccffcc,stroke:#00aa00
    style V fill:#cce6ff,stroke:#0088cc
    style W fill:#ffdddd,stroke:#ff0000
    style G_S fill:#cce6ff,stroke:#0088cc
    style G_P_choice fill:#ffff99,stroke:#ff9900
```
---

## 🗂 6. Database Logging

The backend uses SQLite (chat_history.db) to store:
- User messages
- Agent messages
- Tool data
- Raw JSON results
- Timestamps
- Session IDs
- Stored in the ChatLog table.
- Retrieve history via UI or via API:

```Bash
GET /chat-history/{session_id}
```
---
## 🧩 7. API Endpoints (FastAPI)

| Method | Endpoint                     | Description                             |
| ------ | ---------------------------- | --------------------------------------- |
| GET    | `/`                          | Health check                            |
| POST   | `/research-chat`             | Runs LangGraph and returns final answer |
| GET    | `/graph-visualization`       | Returns PNG of graph structure          |
| GET    | `/chat-history/{session_id}` | Full conversation                       |
| GET    | `/list-sessions`             | All stored sessions                     |

---
## 🖥 8. Frontend (Streamlit UI)
### Features:
### ✔ Chat interface (like ChatGPT)
### ✔ Tool data JSON viewer
### ✔ Workflow graph toggle (PNG from backend)
### ✔ Session switching + history loading
### ✔ Automatic reconnection handling
### ✔ Three-column layout:

| Column | Purpose                                 |
| ------ | --------------------------------------- |
| Left   | History, session loading, system status |
| Middle | Chat interface                          |
| Right  | LangGraph workflow visualization        |

---
## 🧭 9. Graph Visualization

Backend endpoint:
```Bash
GET /graph-visualization
```
The graph is generated via:
```Bash
visualize_graph(research_agent_app)

```

## 🧭 10. CLI Printing for Debugging
<img width="563" height="887" alt="Screenshot 2025-11-30 at 23 05 54" src="https://github.com/user-attachments/assets/b2d5c3d5-9fd2-4789-bd99-7ac7d1b30eba" />








