################################################################################
# PROGRAM NAME: Research Agent Workflow (Backend.py)
# DESCRIPTION:  FastAPI backend for Research Agent. Manages API requests,
#               executes the LangGraph workflow, logs interactions to SQLite,
#               and serves graph visualizations.
# AUTHOR:       AI Agent Team
# DATE:         2025-11-23 (Updated)
# VERSION:      3.1 (NASA-Style Docs & Enhanced Debugging)
# DEPENDENCIES: FastAPI, Pydantic, SQLAlchemy, LangGraph
# CRITICAL:     Ensure 'graph.py' contains 'research_agent_app' and 'visualize_graph'.
################################################################################

import os
import datetime
import re
import uuid
import traceback
import json
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from dotenv import load_dotenv

import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)

# ------------------------------------------------------------------------------
# SECTION 1: MODULE IMPORTS AND CONFIGURATION
# ------------------------------------------------------------------------------
print(" >> [INIT] Loading necessary modules and configuration.")

# Import compiled LangGraph workflow and visualization function
from graph import research_agent_app, visualize_graph

load_dotenv()
OPENAI_API_KEY = os.getenv("GPT_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(" >> [FATAL] Missing OPENAI_API_KEY in environment")
print(" >> [INIT] Environment variables loaded successfully.")

# ------------------------------------------------------------------------------
# SECTION 2: DATABASE SETUP (SQLite)
# ------------------------------------------------------------------------------
################################################################################
# MODULE:       Database Setup
# PURPOSE:      Initialize SQLite database for chat logging.
# LOGIC:        Uses SQLAlchemy ORM.
# CRITICAL:     connect_args={"check_same_thread": False} required for SQLite/FastAPI.
################################################################################
DATABASE_URL = "sqlite:///./chat_history.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


################################################################################
# CLASS:        ChatLog (DB Model)
# PURPOSE:      Store all chat messages and tool outputs for persistence.
# FIELDS:       session_id, role, message, tool_used, raw_data, timestamp.
################################################################################
class ChatLog(Base):
    __tablename__ = "chat_logs"
    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, index=True)
    timestamp = Column(DateTime)
    role = Column(String)
    message = Column(Text)
    tool_used = Column(String)
    raw_data = Column(Text)


Base.metadata.create_all(bind=engine)
print(" >> [INIT] Database structure verified/created.")


# ------------------------------------------------------------------------------
# SECTION 3: API SETUP AND MODELS
# ------------------------------------------------------------------------------
app = FastAPI(title="Research Agent API with SQLite Logging")

################################################################################
# MODULE:       CORS Middleware
# PURPOSE:      Allows the Streamlit frontend (or any external client) to access the API.
################################################################################
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatEntry(BaseModel):
    timestamp: datetime.datetime
    role: str
    message: str
    tool_used: Optional[str] = None
    raw_data: Optional[str] = None


# ------------------------------------------------------------------------------
# SECTION 4: HELPER FUNCTIONS
# ------------------------------------------------------------------------------

################################################################################
# FUNCTION:     log_to_db
# PURPOSE:      Logs messages and tool outputs to the SQLite database.
# INPUTS:       session_id, role (user/agent), message, tool_used, raw_data.
# NOTES:        Uses json.dumps for robust storage of complex dictionary data.
################################################################################
def log_to_db(session_id: str, role: str, message: str, tool_used: Optional[str] = None, raw_data: Any = None):
    db = SessionLocal()
    try:
        # CRITICAL: Ensure complex objects are stringified for DB storage.
        raw_str = json.dumps(raw_data) if isinstance(raw_data, (dict, list)) else str(raw_data)

        db.add(
            ChatLog(
                id=str(uuid.uuid4()),
                session_id=session_id,
                timestamp=datetime.datetime.utcnow(),
                role=role,
                message=message,
                tool_used=tool_used,
                raw_data=raw_str,
            )
        )
        db.commit()
        print(f" >> [DB LOG] Successfully logged message. Role: {role}, Tool: {tool_used}")
    except Exception as e:
        print(f" >> [DB ERROR] Failed to log message: {e}")
    finally:
        db.close()


################################################################################
# FUNCTION:     _cleanse_text_data_ultimate and _cleanse_recursive_state
# PURPOSE:      remove problematic surrogate characters before posting
#               Recursively cleanses all strings within a dictionary or list.
# INPUTS:       session_id, role (user/agent), message, tool_used, raw_data.
# NOTES:        Uses json.dumps for robust storage of complex dictionary data.
################################################################################
def _cleanse_text_data_ultimate(text: str) -> str:
    """The safety function to remove problematic surrogate characters."""
    if not isinstance(text, str):
        return ""
    # Pattern to find and remove Unicode Surrogate Code Points (U+D800 to U+DFFF)
    surrogate_pattern = re.compile(r'[\ud800-\udfff]')
    safe_text = surrogate_pattern.sub('', text)
    try:
        # Use UTF-8 encode/decode to clean up any remaining invalid sequences
        return safe_text.encode('utf-8', 'ignore').decode('utf-8').strip()
    except Exception:
        return safe_text.strip()

def _cleanse_recursive_state(data: Any) -> Any:
    """Recursively cleanses all strings within a dictionary or list."""
    if isinstance(data, str):
        return _cleanse_text_data_ultimate(data)
    elif isinstance(data, list):
        # Apply cleansing to every item in the list
        return [_cleanse_recursive_state(item) for item in data]
    elif isinstance(data, dict):
        # Apply cleansing to all keys and values in the dictionary
        return {k: _cleanse_recursive_state(v) for k, v in data.items()}
    else:
        # Return non-string/list/dict types unchanged
        return data

# ------------------------------------------------------------------------------
# SECTION 5: API ENDPOINTS
# ------------------------------------------------------------------------------

################################################################################
# ENDPOINT:     /
# METHOD:       GET
# PURPOSE:      Health check endpoint.
################################################################################
@app.get("/")
async def home():
    print(" >> [HOME] Health check called.")
    return {
        "status": "running",
        "agent": "v3.1-declarative-state-graph",
        "storage": "SQLite",
    }


################################################################################
# ENDPOINT:     /graph-visualization
# METHOD:       GET
# PURPOSE:      Serves the LangGraph visualization as a PNG image.
# OUTPUT:       Response(content=png_bytes, media_type="image/png")
################################################################################
@app.get("/graph-visualization")
async def get_graph_visualization():
    print(" >> [GRAPH] Request received. Generating visualization...")
    loop = asyncio.get_running_loop()
    try:
        png_bytes = await loop.run_in_executor(executor, visualize_graph, research_agent_app)
        if not png_bytes:
            raise RuntimeError("Graphviz returned empty PNG bytes")
        print(" >> [GRAPH] Visualization generation complete.")
        return Response(content=png_bytes, media_type="image/png")
    except Exception:
        print(f" >> [GRAPH ERROR] {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error generating graph visualization.")


################################################################################
# ENDPOINT:     /research-chat
# METHOD:       POST
# PURPOSE:      Main endpoint for research agent queries. Executes LangGraph workflow.
# INPUT:        Query (session_id, message)
# OUTPUT:       Final state dictionary with answer and raw data.
################################################################################
@app.post("/research-chat")
async def research_chat(q: Query):
    if not q.message:
        print(" >> [CHAT] Empty message received. Aborting.")
        return {"error": "Message cannot be empty"}

    session_id = q.session_id or str(uuid.uuid4())
    print(f" >> [CHAT] Start: Session ID {session_id[:8]}..., Query: '{q.message[:40]}...'")

    # Log user query (Ensure the log function itself is safe if using un-cleansed strings)
    log_to_db(session_id=session_id, role="user", message=q.message)

    try:
        initial_state = {"query": q.message, "cached_results": {}, "context": {}}

        # EXECUTION: Use .invoke() to get the final state synchronously
        print(" >> [AGENT EXEC] Invoking synchronous workflow...")
        result = research_agent_app.invoke(initial_state)
        print(" >> [AGENT EXEC] Workflow finished.")

        # --- CRITICAL FIX: RECURSIVELY CLEANSE THE ENTIRE RESULT PAYLOAD ---
        # This prevents the UnicodeEncodeError from any un-cleansed field (raw_data, context, etc.)
        cleansed_result = _cleanse_recursive_state(result)
        # ------------------------------------------------------------------

        # EXTRACT RESULTS from the CLEANNSED object
        final_answer = cleansed_result.get("answer", "Agent failed to produce a final answer.")
        final_tool = cleansed_result.get("tool", "multi_tool")
        final_raw = cleansed_result.get("raw_data", "No tool data available in final state.")

        # Log agent response (Use the already cleansed data)
        log_to_db(
            session_id=session_id,
            role="agent",
            message=final_answer,
            tool_used=final_tool,
            raw_data=final_raw,
        )
        print(f" >> [CHAT] Success. Final answer generated ({len(final_answer)} chars).")

        # Return the cleansed data, which Starlette/FastAPI can safely serialize
        return {
            "session_id": session_id,
            "response": final_answer,
            "tool_used": final_tool,
            "raw_data": final_raw,
            "aggregated_subtasks": cleansed_result, # Return the full cleansed final state
        }

    except Exception as e:
        print(f" >> [CHAT ERROR] Critical error during workflow execution: {str(e)}")
        error_trace = traceback.format_exc()

        # Log error (Ensure this log is also cleansed for safety)
        clean_error_message = _cleanse_text_data_ultimate(f"Agent crashed during execution: {str(e)}")

        log_to_db(
            session_id=session_id,
            role="agent_error",
            message=clean_error_message,
            tool_used="error_handler",
            raw_data=_cleanse_text_data_ultimate(error_trace), # Cleanse traceback too
        )

        raise HTTPException(
            status_code=500,
            detail={"error": "An internal agent error occurred.", "traceback": error_trace}
        )


################################################################################
# ENDPOINT:     /chat-history/{session_id}
# METHOD:       GET
# PURPOSE:      Retrieves chat history for a specific session ID from the DB.
# OUTPUT:       List of ChatEntry objects.
################################################################################
@app.get("/chat-history/{session_id}", response_model=List[ChatEntry])
async def get_chat_history(session_id: str):
    db = SessionLocal()
    try:
        logs = (
            db.query(ChatLog)
            .filter(ChatLog.session_id == session_id)
            .order_by(ChatLog.timestamp)
            .all()
        )
        print(f" >> [HISTORY] Retrieved {len(logs)} messages for session {session_id[:8]}...")

        return [
            ChatEntry(
                timestamp=log.timestamp,
                role=log.role,
                message=log.message,
                tool_used=log.tool_used,
                raw_data=log.raw_data,
            ) for log in logs
        ]
    except Exception as e:
        print(f" >> [HISTORY ERROR] Failed to fetch history: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error fetching chat history")
    finally:
        db.close()

################################################################################
# ENDPOINT:     /list-sessions
# METHOD:       GET
# PURPOSE:      Returns a list of all unique session IDs with a preview of the
#               last message and timestamp. Useful for frontend chat history panel.
# OUTPUT:       JSON list: [{"session_id": ..., "last_msg": ..., "last_ts": ...}, ...]
################################################################################
@app.get("/list-sessions")
async def list_sessions():
    db = SessionLocal()
    try:
        # Fetch distinct session IDs
        session_ids = db.query(ChatLog.session_id).distinct().all()
        session_list = []

        for (sid,) in session_ids:
            # Fetch the last message for this session
            last_log = (
                db.query(ChatLog)
                .filter(ChatLog.session_id == sid)
                .order_by(ChatLog.timestamp.desc())
                .first()
            )
            if last_log:
                session_list.append({
                    "session_id": sid,
                    "last_msg": last_log.message[:100],  # Preview first 100 chars
                    "last_ts": last_log.timestamp.isoformat()
                })

        print(f" >> [LIST SESSIONS] Retrieved {len(session_list)} sessions.")
        return session_list

    except Exception as e:
        print(f" >> [LIST SESSIONS ERROR] Failed to fetch session list: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error fetching session list")
    finally:
        db.close()