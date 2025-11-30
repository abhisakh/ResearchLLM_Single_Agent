# Frontend (run the ui_main.py which is linked with the graph.py )
```Bash
streamlit run ui_main.py
```
# Backend (run the backend.py)
```Bash
uvicorn backend:app --reload
```

## ✅ High-Level Workflow (with FastAPI backend)
```Bash
User
  ↓
FastAPI /research-chat endpoint
  ↓
LLM (Intent Detection)
  ↓
Decision Layer (LangGraph / Agent Router)
  ↓
Tools (mp_api, arxiv_api, python, vector search)
  ↓
Raw Data
  ↓
Second LLM Pass (Meta-level decision)
  ↓
Should return to user?
   ↳ Yes → Final Answer
   ↳ No → Loop back → more tools or more reasoning
```

## 🧱 Recommended Final Setup
- **User** side (Prod UI):
✔ Agent ChatUI

Clean, polished, modern conversation UI.

- **Developer** side (Internal UI):
✔ Streamlit Admin Dashboard

Includes:

LangGraph visualization

Node-by-node breakdown

Tool output previews

Error logs

Retry buttons

Agent version controls

Test prompts

Data inspection

This is what you need for:

debugging

training new agents

adding tools

observability

Aim:




## We needed

---
✔ Architecture diagram
```Bash
                              ┌──────────────────────────────┐
                              │          User (Web)           │
                              │    Clean Agent ChatUI (Prod)  │
                              └───────────────┬───────────────┘
                                              │ HTTPS
                                              ▼
                               ┌────────────────────────────┐
                               │         FastAPI API        │
                               │   /research-chat endpoint  │
                               └───────────────┬────────────┘
                                               │
                 ┌─────────────────────────────┴────────────────────────────┐
                 │                        LLM Engine                         │
                 │                   (OpenAI / Anthropic)                    │
                 └─────────────────────────────┬────────────────────────────┘
                                               │
                              ┌────────────────┴────────────────┐
                              │        Intent Detection          │
                              └────────────────┬─────────────────┘
                                               │
                               ┌───────────────┴─────────────────┐
                               │     LangGraph Decision Layer     │
                               │  Router → Nodes → Subgraphs     │
                               └───────────────┬─────────────────┘
                                               │
           ┌───────────────────────────────────┼──────────────────────────────────┐
           │                                   │                                  │
┌──────────┴──────────┐             ┌──────────┴─────────┐             ┌──────────┴───────────┐
│   Tools Layer        │             │   Python Node       │             │  Vector Store / RAG   │
│ • mp_api             │             │ • Data cleaning      │             │ • FAISS / LanceDB     │
│ • arxiv_api          │             │ • Computation        │             │ • Similarity search   │
│ • Web search         │             │ • Graph building     │             └───────────────────────┘
│ • Local DB           │             └──────────────────────┘
└──────────────────────┘
                                               │
                               ┌───────────────┴───────────────────┐
                               │     Second LLM Pass (Refiner)     │
                               │  “Should I answer or continue?”   │
                               └───────────────┬───────────────────┘
                                               │
                    Yes ───────────────────────┘       └──────→ No → back into LangGraph
                                               ▼
                                 ┌──────────────────────────┐
                                 │      Final Response      │
                                 │  (FastAPI → ChatUI)      │
                                 └──────────────────────────┘

────────────────────────────────────────────────────────────────────────
Parallel Developer-Side Interface:
────────────────────────────────────────────────────────────────────────

                                   Streamlit Admin Dashboard
────────────────────────────────────────────────────────────────────────
• LangGraph visualization
• Node-by-node output
• Live logs and tool traces
• Retry last step
• Edit agent config
• Test prompts
• Inspect API responses


```
---
✔ Streamlit code for LangGraph visualization

---
✔ Agent ChatUI integration with your FastAPI

---
✔ Dual-UI folder/project structure

---
✔ Example LangGraph debugging interface

---




## 🟪 Why Agent ChatUI Is Not Ideal right now

Agent ChatUI is:

✔ Beautiful
✔ Professional
✔ Great for a polished ChatGPT-like UX

But it is mainly designed for:

agent-tool execution view

conversation

tool call visualization

It is not designed for:

data tables

plots

materials dashboards

research logs

workflow visualizations

You can use it, but you would lose flexibility.






## 🧠 Why a Second LLM Pass Is Important

- This second reasoning step enables:

✔ Verification

- LLM checks if the retrieved data is sufficient, consistent, and relevant.

✔ Multi-step research reasoning

Example:

- - User: “Find best perovskite for solar cells.”

- LLM: “Need band gap + stability data.”
→ Tools used
→ LLM checks if enough data exists
→ Might say: “Need additional papers from arXiv for comparison.”

✔ Safety / Reliability

- LLM decides if:

- - Results are incomplete

- - Conflicting data exists

- - More querying needed

Or final answer is safe + ready for user

✔ Fully autonomous research chain

- This is closer to “mini-research agent” rather than a search engine.

## 🧱 Recommended Architecture With FastAPI + Streamlit
```Bash
Streamlit UI
    ↓
FastAPI Backend
    ↓
LangGraph Agent (Recursive)
    ↓
LLM ↔ Tools ↔ LLM
    ↓
Final Answer
```


## 🌟 Why the Second LLM Pass Is the Key to Future Extensions

When the agent finishes retrieving data (from mp_api, arxiv_api, etc.), and that raw data flows back into the LLM, you unlock the ability to:

### 🔮 1. Apply Custom Logic in Future Versions
Example extensions:

Screening materials using domain-specific rules

Running ML models (band gap prediction, crystal stability models)

Multi-criteria optimization (Pareto-front)

Suggesting synthesis methods based on research trends

Running automated literature meta-analysis

Re-evaluating correctness or relevance

Asking follow-up questions automatically

Because the LLM sees the full intermediate data, it can reason about:

What is missing

What is inconsistent

What needs deeper analysis

Whether another tool is needed

### 🔬 2. Build a Fully Autonomous Research Agent (Future Goal)

This architecture is ideal if you want to eventually build a:

“Materials Auto-Research Agent”

that can:

Fetch data

Check quality

Analyze patterns

Decide next steps

Iterate without human prompts

The second-pass LLM allows multi-step reasoning like:

“Band gap is available, but stability data is missing → call tool B.”

“Literature says A, database says B → need cross-validation.”

“Two papers contradict — summarize differences.”

This is only possible because the LLM is given the raw tool outputs for deeper reasoning.

### ⚙️ 3. Add Future Data Processing Modules Before the Second Pass

You can insert any future module between the tool outputs and the second LLM pass, for example:

Possible future modules:

Material-property calculators

Machine-learning prediction models

Phase diagram solvers

Crystallographic analysis

DFT data post-processing

Data validation pipeline

All these modules can be added WITHOUT changing the frontend or the decision logic — only plug into the pipeline before LLM pass 2.

#### 🔁 4. Multi-Step Loops Become Natural

With this design, the LLM can do iterative reasoning:
```Bash
LLM → Tools → Data → LLM → Tools → Data → LLM → … → Final Answer
```

This future-proofs your system for:

Multi-hop scientific reasoning

Iterative querying

Long research workflows

LangGraph supports loop nodes, which make this trivial.

### 🧠 5. Advanced Behaviors Become Possible Later:
✔ Data consistency checking
✔ Knowledge-graph creation
✔ Workflow planning
✔ Materials screening pipelines
✔ Experiment planning
✔ Automated hypothesis generation

The key is:
LLM needs full access to tool results, history, and context to make smart decisions.

Your architecture supports this perfectly.

## 🎯 Why Streamlit Is the Better Choice at This Stage

Since your system involves:

FastAPI backend

LangGraph / Agent reasoning loops

Second-pass LLM logic

Research data tables

Material properties visualization

Graphs and charts

Ability to extend with future scientific modules

Streamlit gives you all of this easily, without touching JavaScript.

✔ Best for rapid development

You will iterate fast.

✔ Python-only

No React, no HTML, no JS — perfect for scientific workflows.

✔ Built-in charts, tables, dataframe viewer

Great for material properties, band gap plots, etc.

✔ Easy integration with FastAPI

Streamlit → HTTP → FastAPI → Agent → Tools → LLM
Smooth and simple.

✔ Perfect for scientific dashboards

You can show:

tables of mp_api results

plots (matplotlib, plotly)

PDF/abstract previews

structure information

✔ Ideal while the system is evolving

As you build:

decision loops

new tools

custom processors
Streamlit adapts easily.

## ✅ 1. User-Facing UI

Clean

Simple

Only the final results

No agent graph

No internal reasoning

Professional interface

This could be:

👉 Agent ChatUI (Recommended for users)

A beautiful user-facing chat interface with streaming, tool-call visualization, etc.

OR

👉 Streamlit Light Mode

If you want a dashboard-style research interface.

## ✅ 2. Developer-Facing UI

This is where you:

Debug agents

Visualize LangGraph

Inspect control flow

Add tools or sub-agents

View tool outputs

Examine internal steps

See tokens, messages, errors

Run “dry-run” modes

For this developer UI, the best tool is Streamlit or a custom FastAPI Admin panel.

And LangGraph natively supports graph visualization, so you can plug it into Streamlit or a local admin dashboard.

```Bash
                    ┌──────────────────────────────┐
                    │      Developer Interface      │
                    │  (Streamlit + LangGraph viz)  │
                    └──────────────────────────────┘
                                  │
             ┌───────────────────┴──────────────────┐
             │                                      │
┌────────────────────────┐           ┌────────────────────────────┐
│   User Interface        │           │   FastAPI Backend          │
│ (Agent ChatUI / Streamlit)│        │ (Router + LangGraph Agent) │
└────────────────────────┘           └────────────────────────────┘
                                                 │
                                   ┌─────────────┴─────────────┐
                                   │         Tools               │
                                   │ (mp_api, arxiv, python...) │
                                   └─────────────────────────────┘

```
## ====================================================================
✅ 1. AGENT (graph.py)
## ====================================================================

- (1) Intent Agent → maps query to tool category
- (2) Tool-Decision Router → selects the correct tool
- (3) Tools Layer → actual API calls
- (4) Meta-Agent → continue vs return

🎉 Our agent now supports MULTIPLE REAL TOOLS!
✔ arXiv search
✔ Materials Project (Chemistry + Materials Science)
✔ Vector Search via FAISS
✔ Python compute
✔ LLM-based intent detection
✔ LLM-based meta-decision
✔ Automatic looping
✔ Fully integrated LangGraph flow

## ============================================================
✅ 2. Updated ui_main.py (Frontend Chat UI)
- - To Run # streamlit run ui_main.py
## ============================================================
✔ Clean CLI UI
✔ Shows which tool the agent selected
✔ Shows errors
✔ Pretty formatting

## ============================================================
✅ 3. Updated ui_admin.py (Streamlit Admin Dashboard)
## ============================================================
✔ Full graph debugging view
✔ Shows tool selection
✔ Shows raw tool output
✔ Shows meta-reasoning
✔ Real-time trace viewer
✔ Beautiful layout

## ============================================================
📦 4. Backend (FINAL) + SQL log

- - To Run # uvicorn backend:app --reload
## ============================================================
✅ Key Features

- Session-based logging: Each chat belongs to a session_id.
- Timestamps: Every message logged with UTC timestamp.
- Role tracking: user vs agent.
- Tool & Raw Data: Tools used and raw outputs saved.
- History endpoint: /history/{session_id} to fetch full chat logs.
- Automatic UUID generation if session not provided.

## ============================================================
📦 5. Updated Requirements (FINAL)
## ============================================================
fastapi
uvicorn
openai>=1.0.0
langgraph
langchain-community
streamlit
streamlit-json
requests
pydantic
python-dotenv

# Tools
arxiv
mp-api
faiss-cpu
numpy

# database
sqlalchemy
databases

# =============================================================
🎉 DONE — All components fully updated!
Your system now supports:
🤖 Multi-Tool Research Agent (LangGraph)

arXiv search

Materials Project

Python compute

Vector search (FAISS)

🔄 Multi-step workflow

Intent agent

Tool router

Tool execution

Data analyzer

Meta-agent

Loop until complete

Final answer agent

🖥 Two UIs

Clean CLI chat

Full admin/debug dashboard

🚀 REST API Backend

Perfect for production

Trace-enabled

CORS-enabled

👉 WANT MORE?


🔹 A React web UI for the user chat front-end
🔹 A Docker Compose setup
🔹 A PostgreSQL/Redis memory store
🔹 A vector DB integration (LanceDB / Pinecone)


# To run Backend
```Bash
uvicorn backend:app --reload
```

# To run Frontend
```Bash
streamlit run ui_main.py
```