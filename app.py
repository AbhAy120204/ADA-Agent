"""
Streamlit UI for the Autonomous Data Analyst Agent.

Key Streamlit concepts used here:
  st.session_state  — persists values across reruns (Streamlit reruns the whole
                      script on every interaction, so without this, results vanish)
  st.empty()        — a placeholder that can be overwritten in-place (used for
                      live updates while the agent is running)
  graph.stream()    — yields state after each node, which we render incrementally
"""

import tempfile
import os
import streamlit as st
import pandas as pd

from agent.graph import stream_analysis

# ── Step renderer ─────────────────────────────────────────────────────────────

def _render_step(node: str, state: dict) -> None:
    """
    Renders a single agent step as a styled card.
    Each node type gets a different icon so the user can
    follow the ReAct loop visually.
    """
    NODE_CONFIG = {
        "planner":     ("🧠", "Plan"),
        "code_gen":    ("✍️",  "Code written"),
        "executor":    ("⚙️",  "Executed"),
        "error_fixer": ("🔧", "Fixing error"),
        "reflector":   ("🪞", "Insight"),
        "summarizer":  ("📋", "Summary"),
    }

    icon, label = NODE_CONFIG.get(node, ("▶️", node))

    with st.expander(f"{icon} {label}", expanded=(node not in ("summarizer",))):
        if node == "planner" and state.get("current_plan"):
            st.markdown(f"**Next question to investigate:**\n\n{state['current_plan']}")

        elif node == "code_gen" and state.get("current_code"):
            st.code(state["current_code"], language="python")

        elif node == "executor" and state.get("execution_result"):
            result = state["execution_result"]
            if result.startswith("ERROR:"):
                st.error(result)
            else:
                st.code(result, language="text")

        elif node == "error_fixer":
            st.warning(f"Retry attempt {state.get('error_count', '?')}/3")
            if state.get("current_code"):
                st.code(state["current_code"], language="python")

        elif node == "reflector" and state.get("insights"):
            latest = state["insights"][-1]
            if latest.startswith("INSIGHT"):
                st.success(latest)
            else:
                st.warning(latest)

        elif node == "summarizer" and state.get("final_summary"):
            st.markdown(state["final_summary"])


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Autonomous Data Analyst",
    page_icon="🔍",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com",
    )
    # Fall back to .env if the user didn't type a key in the sidebar
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY", "")

    max_iterations = st.slider(
        "Analysis depth (iterations)",
        min_value=2,
        max_value=8,
        value=5,
        help="Each iteration = one analysis question the agent investigates",
    )

    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Upload a CSV\n"
        "2. Agent plans → writes code → runs it → reflects\n"
        "3. Loop repeats, building insights\n"
        "4. Final executive summary generated"
    )
    st.divider()
    st.caption("Built with LangGraph · Groq · Streamlit")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🔍 Autonomous Data Analyst Agent")
st.caption("Upload a CSV. The agent writes code, runs it, fixes errors, and surfaces insights — automatically.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Preview the data so the user knows what was loaded
    df_preview = pd.read_csv(uploaded_file)
    uploaded_file.seek(0)  # reset so it can be read again when we save to disk

    with st.expander(f"📋 Data preview — {df_preview.shape[0]} rows × {df_preview.shape[1]} columns", expanded=True):
        st.dataframe(df_preview.head(10), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df_preview.shape[0])
        col2.metric("Columns", df_preview.shape[1])
        col3.metric("Null cells", int(df_preview.isnull().sum().sum()))

    st.divider()

    run_clicked = st.button("🚀 Run Analysis", type="primary", disabled=not api_key)
    if not api_key:
        st.warning("Enter your Groq API key in the sidebar to run the analysis.")

    if run_clicked and api_key:
        # Save upload to a temp file so our tools.py can read it from disk
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Clear previous run results from session state
        st.session_state["final_summary"] = ""
        st.session_state["insights"] = []

        # st.status() is the "Agent thinking..." collapsible container.
        # - While the agent runs it shows a spinner and stays expanded.
        # - When done it collapses with a green checkmark — same UX as ChatGPT's
        #   "Searched X sources" or Claude's "Thinking" dropdown.
        # - User can click it anytime to watch the internal steps.
        with st.status("🤖 Agent thinking...", expanded=True) as status:
            try:
                for event in stream_analysis(tmp_path, max_iterations=max_iterations, api_key=api_key):
                    node = event["node"]
                    state = event["state"]

                    _render_step(node, state)

                    # Capture final state values
                    if node == "summarizer":
                        st.session_state["final_summary"] = state.get("final_summary", "")
                    if state.get("insights"):
                        st.session_state["insights"] = state["insights"]

                status.update(label="✅ Analysis complete", state="complete", expanded=False)

            except Exception as e:
                status.update(label="❌ Agent error", state="error", expanded=True)
                st.error(f"{e}")
            finally:
                os.unlink(tmp_path)

        # ── Final summary — shown outside the collapsible ──────────────────
        if st.session_state.get("final_summary"):
            st.divider()
            st.subheader("📊 Executive Summary")
            st.markdown(st.session_state["final_summary"])

        if st.session_state.get("insights"):
            st.divider()
            st.subheader("💡 All Insights")
            for i, insight in enumerate(st.session_state["insights"], 1):
                icon = "✅" if insight.startswith("INSIGHT") else "⚠️"
                st.markdown(f"{icon} **{i}.** {insight}")
