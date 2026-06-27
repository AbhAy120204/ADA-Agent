"""
Streamlit UI for the Autonomous Data Analyst Agent.

Key Streamlit concepts used here:
  st.session_state  — persists values across reruns
  st.status()       — collapsible "thinking" panel with spinner
  st.plotly_chart() — renders Plotly JSON as interactive charts
  graph.stream()    — yields state after each node for live updates
"""

import json
import tempfile
import os
import streamlit as st
import pandas as pd
import plotly.io as pio

from agent.graph import stream_analysis


def _setup_langsmith(api_key: str, project: str) -> None:
    """Set LangSmith env vars so LangChain auto-traces the run."""
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"


def _teardown_langsmith() -> None:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


# ── Step renderer ─────────────────────────────────────────────────────────────

def _render_step(node: str, state: dict) -> None:
    """Renders one agent step inside the st.status() thinking panel."""
    NODE_CONFIG = {
        "sanitizer":       ("🔬", "Data Quality Check"),
        "sanitizer_store": ("📋", "Data Quality Report"),
        "planner":         ("🧠", "Plan"),
        "code_gen":        ("✍️",  "Code written"),
        "executor":        ("⚙️",  "Executed"),
        "error_fixer":     ("🔧", "Fixing error"),
        "reflector":       ("🪞", "Insight"),
        "summarizer":      ("📋", "Summary"),
    }
    icon, label = NODE_CONFIG.get(node, ("▶️", node))

    with st.expander(f"{icon} {label}", expanded=True):
        if node == "sanitizer":
            st.info("Running data quality inspection...")

        elif node == "sanitizer_store":
            report = state.get("data_quality_report", "")
            if report:
                lines = report.splitlines()
                high   = sum(1 for l in lines if "[HIGH]"   in l)
                medium = sum(1 for l in lines if "[MEDIUM]" in l)
                low    = sum(1 for l in lines if "[LOW]"    in l)
                c1, c2, c3 = st.columns(3)
                c1.metric("🔴 High",   high)
                c2.metric("🟡 Medium", medium)
                c3.metric("🔵 Low",    low)
                with st.expander("View full report", expanded=False):
                    st.code(report, language="text")
            else:
                st.success("No data quality issues detected.")

        elif node == "planner" and state.get("current_plan"):
            st.markdown(f"**Next question:**\n\n{state['current_plan']}")

        elif node == "code_gen" and state.get("current_code"):
            st.code(state["current_code"], language="python")

        elif node == "executor" and state.get("execution_result"):
            result = state["execution_result"]
            if result.startswith("ERROR:"):
                st.error(result)
            else:
                st.code(result, language="text")
            # Note: charts are NOT rendered here — only in the gallery below.
            # st.plotly_chart inside st.status() causes Streamlit rendering errors.

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

    PROVIDER_CONFIG = {
        "Groq": {
            "id": "groq",
            "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            "key_placeholder": "gsk_...",
            "key_help": "Get a free key at console.groq.com",
            "env_var": "GROQ_API_KEY",
        },
        "Gemini": {
            "id": "gemini",
            "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
            "key_placeholder": "AIza...",
            "key_help": "Get a free key at aistudio.google.com",
            "env_var": "GOOGLE_API_KEY",
        },
    }

    provider_label = st.selectbox("Provider", list(PROVIDER_CONFIG.keys()), index=0)
    provider_cfg = PROVIDER_CONFIG[provider_label]
    provider = provider_cfg["id"]

    api_key = st.text_input(
        f"{provider_label} API Key",
        type="password",
        placeholder=provider_cfg["key_placeholder"],
        help=provider_cfg["key_help"],
    )
    if not api_key:
        api_key = os.getenv(provider_cfg["env_var"], "")

    model = st.selectbox(
        "Model",
        provider_cfg["models"],
        index=0,
        help="Smaller models are faster and cheaper; larger ones produce better analysis.",
    )

    max_iterations = st.slider(
        "Analysis depth (iterations)",
        min_value=2,
        max_value=8,
        value=5,
        help="Each iteration = one analysis question the agent investigates",
    )

    st.divider()
    with st.expander("🔭 LangSmith Tracing (optional)"):
        langsmith_key = st.text_input(
            "LangSmith API Key",
            type="password",
            placeholder="lsv2_...",
            help="Get a free key at smith.langchain.com",
        )
        langsmith_project = st.text_input(
            "Project name",
            value="ada-agent",
            help="Traces will appear under this project in LangSmith",
        )
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Upload a CSV\n"
        "2. Agent plans → writes code → runs it → reflects\n"
        "3. Loop repeats, building insights\n"
        "4. Charts generated automatically\n"
        "5. Final executive summary produced"
    )
    st.divider()
    st.caption("Built with LangGraph · Groq · Streamlit")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🔍 Autonomous Data Analyst Agent")
st.caption("Upload a CSV. The agent writes code, runs it, fixes errors, generates charts, and surfaces insights — automatically.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df_preview = pd.read_csv(uploaded_file)
    uploaded_file.seek(0)

    with st.expander(f"📋 Data preview — {df_preview.shape[0]} rows × {df_preview.shape[1]} columns", expanded=True):
        st.dataframe(df_preview.head(10), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df_preview.shape[0])
        col2.metric("Columns", df_preview.shape[1])
        col3.metric("Null cells", int(df_preview.isnull().sum().sum()))

    st.divider()

    run_clicked = st.button("🚀 Run Analysis", type="primary", disabled=not api_key)
    if not api_key:
        st.warning(f"Enter your {provider_label} API key in the sidebar to run the analysis.")

    if run_clicked and api_key:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.session_state["final_summary"] = ""
        st.session_state["insights"] = []
        st.session_state["charts"] = []
        st.session_state["token_usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        st.session_state["data_quality_report"] = ""

        if langsmith_key:
            _setup_langsmith(langsmith_key, langsmith_project)

        with st.status("🤖 Agent thinking...", expanded=True) as status:
            try:
                for event in stream_analysis(
                    tmp_path,
                    max_iterations=max_iterations,
                    provider=provider,
                    api_key=api_key,
                    model=model,
                ):
                    node = event["node"]
                    state = event["state"]

                    _render_step(node, state)

                    if node == "summarizer":
                        st.session_state["final_summary"] = state.get("final_summary", "")
                    if state.get("insights"):
                        st.session_state["insights"] = state["insights"]
                    if state.get("charts"):
                        st.session_state["charts"] = state["charts"]
                    if state.get("token_usage"):
                        st.session_state["token_usage"] = state["token_usage"]
                    if node == "sanitizer_store" and state.get("data_quality_report"):
                        st.session_state["data_quality_report"] = state["data_quality_report"]

                status.update(label="✅ Analysis complete", state="complete", expanded=False)

            except Exception as e:
                status.update(label="❌ Agent error", state="error", expanded=True)
                st.error(f"{e}")
            finally:
                os.unlink(tmp_path)
                if langsmith_key:
                    _teardown_langsmith()

        # ── Token usage ───────────────────────────────────────────────────
        usage = st.session_state.get("token_usage", {})
        if usage.get("total_tokens"):
            st.divider()
            st.subheader("🔢 Token Usage")
            c1, c2, c3 = st.columns(3)
            c1.metric("Prompt tokens",     f"{usage['prompt_tokens']:,}")
            c2.metric("Completion tokens", f"{usage['completion_tokens']:,}")
            c3.metric("Total tokens",      f"{usage['total_tokens']:,}")

        # ── LangSmith trace link ───────────────────────────────────────────
        if langsmith_key:
            st.info(
                f"🔭 Traces available in LangSmith project **{langsmith_project}** — "
                "[open dashboard](https://smith.langchain.com)",
                icon="🔗",
            )

        # ── Results outside the collapsible ───────────────────────────────
        if st.session_state.get("final_summary"):
            st.divider()
            st.subheader("📊 Executive Summary")
            st.markdown(st.session_state["final_summary"])

        # Chart gallery — all charts produced during the run
        all_charts = [c for c in st.session_state.get("charts", []) if c]
        if all_charts:
            st.divider()
            st.subheader("📈 Charts")
            cols = st.columns(min(len(all_charts), 2))
            for i, chart_json in enumerate(all_charts):
                with cols[i % 2]:
                    fig = pio.from_json(chart_json)
                    st.plotly_chart(fig, use_container_width=True)

        if st.session_state.get("insights"):
            st.divider()
            st.subheader("💡 All Insights")
            for i, insight in enumerate(st.session_state["insights"], 1):
                icon = "✅" if insight.startswith("INSIGHT") else "⚠️"
                st.markdown(f"{icon} **{i}.** {insight}")
