################################################################################
# FILE: main_ui.py
# PURPOSE: Streamlit frontend for Declarative Research Agent
# FEATURES:
#   - Interactive chat with research agent
#   - Robust JSON parsing for tool outputs
#   - Graph visualization toggle
#   - Handles backend connection errors gracefully
#   - Multi-session chat history retrieval
#   - Three-column layout (Left: sessions/history, Middle: chat, Right: LangGraph)
# AUTHOR: AI Agent Team
# DATE: 2025-11-26
# VERSION: 3.3-NASA-Style-UI
# DEPENDENCIES: Streamlit, Requests, UUID, JSON
################################################################################

import streamlit as st
import requests
import json
import uuid

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
API_BASE_URL = "http://localhost:8000"

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------

def get_session_id():
    """Retrieve or initialize the Streamlit session ID."""
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = str(uuid.uuid4())
    return st.session_state['session_id']

def reset_chat_session():
    """Reset chat and session state."""
    st.session_state['session_id'] = str(uuid.uuid4())
    st.session_state['messages'] = []
    st.session_state['show_graph'] = False
    st.session_state['rerun_trigger'] = str(uuid.uuid4())  # Trigger UI refresh

def safe_json_format(data):
    """Safely formats data as a JSON string or returns plain text."""
    try:
        if isinstance(data, str):
            parsed = json.loads(data)
            return json.dumps(parsed, indent=2)
        return json.dumps(data, indent=2)
    except (TypeError, json.JSONDecodeError):
        return str(data)

def fetch_history(session_id: str):
    """
    Fetch chat history from backend and populate Streamlit session.

    Args:
        session_id (str): The session ID to fetch.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/chat-history/{session_id}")
        response.raise_for_status()
        history = response.json()
        messages = []

        for entry in history:
            role = "user" if entry['role'] == 'user' else "assistant"
            message_text = entry['message']

            raw = entry.get('raw_data')
            if raw and raw != "None" and raw != "{}":
                message_text += f"\n\n**🔬 Tool Data:**\n```json\n{safe_json_format(raw)}\n```"

            messages.append({"role": role, "content": message_text})

        st.session_state['messages'] = messages

    except requests.exceptions.RequestException:
        st.session_state['messages'] = []

def display_graph_visualization():
    """
    Fetch and display LangGraph workflow visualization from backend.
    If the backend returns an error, show appropriate error message.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/graph-visualization")
        if response.status_code == 200:
            st.image(response.content, caption="Agent Workflow", use_column_width=True)
        else:
            st.error("Graph endpoint returned an error.")
    except requests.exceptions.RequestException:
        st.error("Could not connect to backend for graph visualization.")

def fetch_session_list():
    """
    Fetch all available session IDs from backend for selection.

    Returns:
        list[str]: List of session IDs or empty list if error occurs.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/list-sessions")
        response.raise_for_status()
        sessions_raw = response.json()
        return [s['session_id'] for s in sessions_raw] if sessions_raw else []
    except requests.exceptions.RequestException:
        return []

# -----------------------------------------------------------------------------
# STREAMLIT UI SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Research Agent", layout="wide")  # Wide layout for three columns
st.title("🔬 Declarative Research Agent")
st.caption(
    "Autonomous agent utilizing ArXiv, Materials Project, PubMed, and Vector Search."
)

# -----------------------------------------------------------------------------
# INITIALIZE SESSION STATE
# -----------------------------------------------------------------------------
session_id = get_session_id()

if 'messages' not in st.session_state:
    st.session_state['messages'] = []
    fetch_history(session_id)

if 'show_graph' not in st.session_state:
    st.session_state['show_graph'] = False

if 'rerun_trigger' not in st.session_state:
    st.session_state['rerun_trigger'] = str(uuid.uuid4())

# -----------------------------------------------------------------------------
# THREE-COLUMN LAYOUT
# -----------------------------------------------------------------------------
col_left, col_middle, col_right = st.columns([1, 2, 1])

# -----------------------------------------------------------------------------
# LEFT COLUMN: SESSION & HISTORY CONTROLS
# -----------------------------------------------------------------------------
with col_left:
    st.header("Controls & Chat History")
    st.info(f"Current Session: `{session_id[:8]}...`")

    if st.button("🔄 Reset Chat", use_container_width=True):
        reset_chat_session()
        st.experimental_rerun()

    st.divider()

    st.subheader("System Status")
    try:
        response = requests.get(f"{API_BASE_URL}")
        if response.status_code == 200:
            data = response.json()
            st.success(f"Backend: Online ({data.get('agent')})")
        else:
            st.warning("Backend: Error")
    except requests.exceptions.ConnectionError:
        st.error("Backend: Offline")
        st.caption("Please run `uvicorn backend:app ...`")

    st.divider()

    st.subheader("Load Chat History")

    other_session_id = st.text_input(
        "Enter Session ID to load history",
        value="",
        placeholder="Paste existing session ID here..."
    )

    if st.button("📥 Load History for Session", key="load_history"):
        if other_session_id:
            fetch_history(other_session_id)
            st.session_state['rerun_trigger'] = str(uuid.uuid4())

    st.caption("OR select from existing sessions:")
    sessions = fetch_session_list()
    if sessions:
        selected = st.selectbox("Select Session", options=sessions, index=0)
        if st.button("Load Selected Session", key="load_selected"):
            fetch_history(selected)
            st.session_state['rerun_trigger'] = str(uuid.uuid4())

# -----------------------------------------------------------------------------
# MIDDLE COLUMN: CHAT INTERFACE
# -----------------------------------------------------------------------------
with col_middle:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Enter research query (e.g., 'Find papers on Perovskites')..."):

        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state['messages'].append({"role": "user", "content": prompt})

        # Call backend
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🕵️ *Agent is researching...*")

            try:
                payload = {"session_id": session_id, "message": prompt}
                response = requests.post(f"{API_BASE_URL}/research-chat", json=payload)
                response.raise_for_status()
                result = response.json()

                # Extract main response
                final_answer = result.get('response', 'No response text provided.')

                # Extract tool data
                raw_data = result.get('raw_data')
                tool_used = result.get('tool_used')

                # Construct full display message
                display_text = final_answer
                if raw_data and raw_data != "No tool data available in final state.":
                    display_text += f"\n\n---\n**🛠️ Tool Used:** `{tool_used}`\n"
                    display_text += f"**📊 Raw Data:**\n```json\n{safe_json_format(raw_data)}\n```"

                message_placeholder.markdown(display_text)
                st.session_state['messages'].append({"role": "assistant", "content": display_text})

            except requests.exceptions.ConnectionError:
                message_placeholder.error("❌ Connection Error: Is `backend.py` running?")
            except requests.exceptions.HTTPError as e:
                message_placeholder.error(f"❌ Server Error: {e}")
            except Exception as e:
                message_placeholder.error(f"❌ Unexpected Error: {e}")

# -----------------------------------------------------------------------------
# RIGHT COLUMN: LANGRAPH VISUALIZATION
# -----------------------------------------------------------------------------
with col_right:
    st.header("Agent Workflow Graph")
    if st.button("🗺️ Toggle Workflow Graph", use_container_width=True):
        st.session_state['show_graph'] = not st.session_state['show_graph']

    if st.session_state.get('show_graph'):
        display_graph_visualization()
