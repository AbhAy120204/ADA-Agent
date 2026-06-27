"""
The ReAct loop implemented as a LangGraph StateGraph.

Node flow:
  sanitizer ──→ code_gen → executor ──success──→ sanitizer_store ──→ planner
                                └──error──→ error_fixer → executor (retry, max 3x)
                                                  └──3rd failure──→ sanitizer_store

  planner → code_gen → executor ──success──→ reflector → (planner OR summarizer)
                            └──error──→ error_fixer → executor (retry, max 3x)
                                              └──3rd failure──→ reflector anyway

sanitizer: zero-LLM task-setter. Writes a fixed DQ inspection task into
  current_plan and sets phase="sanitizing". code_gen/executor/error_fixer
  are reused 100%. sanitizer_store captures the result into data_quality_report
  and flips phase to "analyzing" before handing off to planner.
  after_executor reads phase to route sanitizing runs to sanitizer_store
  instead of reflector.
"""

from typing import TypedDict, Annotated
import operator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from agent.tools import load_csv, run_python_code


# ── State definition ──────────────────────────────────────────────────────────
# TypedDict tells LangGraph exactly what fields live in the state.
# Annotated[list, operator.add] means "append to this list" instead of replacing it.

class TokenUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AgentState(TypedDict):
    file_path: str                              # CSV path provided by user
    data_summary: str                           # Output of load_csv()
    current_plan: str                           # What the planner decided to explore next
    current_code: str                           # Code written by code_gen
    execution_result: str                       # Output of running the code
    insights: Annotated[list[str], operator.add]     # Accumulated findings (appends each loop)
    charts: Annotated[list[str], operator.add]       # Plotly JSON strings, one per chart produced
    iteration: int                                   # How many ReAct loops we've done
    max_iterations: int                              # Safety limit to prevent infinite loops
    final_summary: str                               # Executive summary produced at the end
    # Phase 2 additions
    error_count: int                                 # Retry attempts for the current task (resets each plan)
    last_error: str                                  # Traceback from the last failed execution
    # Phase 6 additions
    token_usage: TokenUsage                          # Cumulative token counts across all LLM calls
    # Phase 8 additions
    data_quality_report: str                         # Output of sanitizer run — injected into all downstream prompts
    phase: str                                       # "sanitizing" | "analyzing" — controls executor routing


# ── LLM setup ─────────────────────────────────────────────────────────────────

# Default model per provider — can be overridden from the UI
DEFAULT_MODELS = {
    "groq":   "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
}


def _get_llm(provider: str = "groq", api_key: str | None = None, model: str | None = None) -> BaseChatModel:
    """
    Build a LangChain chat model for the chosen provider.

    Lazy imports keep the heavy provider SDKs optional — Streamlit Cloud only
    needs whichever provider the user actually picks.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    model = model or DEFAULT_MODELS.get(provider)

    if provider == "groq":
        from langchain_groq import ChatGroq
        key = api_key or os.getenv("GROQ_API_KEY")
        return ChatGroq(model=model, temperature=0, api_key=key)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = api_key or os.getenv("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=key)

    raise ValueError(f"Unknown provider: {provider!r}. Supported: groq, gemini.")


# Module-level config so all nodes in a run share the same provider/key/model
_runtime_config: dict = {"provider": "groq", "api_key": None, "model": None}


def set_llm_config(provider: str, api_key: str | None, model: str | None = None) -> None:
    """Called by Streamlit before starting a run."""
    _runtime_config.update({"provider": provider, "api_key": api_key, "model": model})


def _llm() -> BaseChatModel:
    """Shorthand used by all nodes — picks up the runtime config."""
    return _get_llm(**_runtime_config)


def _add_tokens(current: TokenUsage, response) -> TokenUsage:
    """Accumulate token counts from a LangChain response's usage_metadata."""
    meta = getattr(response, "usage_metadata", None) or {}
    return {
        "prompt_tokens":     current["prompt_tokens"]     + meta.get("input_tokens", 0),
        "completion_tokens": current["completion_tokens"] + meta.get("output_tokens", 0),
        "total_tokens":      current["total_tokens"]      + meta.get("total_tokens", 0),
    }


# ── Node 0: Sanitizer ────────────────────────────────────────────────────────

_SANITIZER_PLAN = (
    "DATA QUALITY CHECK — inspect the DataFrame and print a structured report covering:\n"
    "1. Null/missing values per column (count + % of rows)\n"
    "2. Mixed-type columns where numeric parsing fails "
    "(currency symbols, text like 'ten', Excel errors '#VALUE!', '#REF!')\n"
    "3. Negative values in columns that should be non-negative\n"
    "4. Statistical outliers: Z-score > 3, report count and worst value per numeric column\n"
    "5. columns outside sensible range then what the heading suggests\n"
    "6. Dirty categoricals: leading/trailing spaces, ALL-CAPS variants, "
    "Excel error strings, near-duplicate names\n"
    "Print [HIGH], [MEDIUM], or [LOW] before each finding. Skip clean columns. "
    "End with a SUMMARY paragraph. "
    "IMPORTANT: always use errors='coerce' with pd.to_numeric(); "
    "always dropna() and guard with isinstance(v, str) before string ops on object columns."
)


def sanitizer_node(state: AgentState) -> dict:
    """
    Zero-LLM task-setter. Injects the fixed DQ inspection task into current_plan
    and marks phase='sanitizing'. code_gen, executor, and error_fixer are all
    reused unchanged. sanitizer_store_node captures the result afterward.
    """
    print("\n[SANITIZER] Queuing data quality inspection task...")
    return {
        "current_plan": _SANITIZER_PLAN,
        "phase": "sanitizing",
        "error_count": 0,
        "last_error": "",
    }


# ── Node 0b: Sanitizer Store ──────────────────────────────────────────────────

def sanitizer_store_node(state: AgentState) -> dict:
    """
    Runs after executor finishes the DQ inspection (phase='sanitizing').
    Copies execution_result → data_quality_report and flips phase → 'analyzing'
    so after_executor routes to reflector for all subsequent iterations.
    """
    report = state.get("execution_result", "")
    print(f"\n[SANITIZER STORE] Report captured ({len(report)} chars). Handing off to planner.")
    return {
        "data_quality_report": report,
        "phase": "analyzing",
    }


# ── Node 1: Planner ───────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> dict:
    """
    Looks at what we know so far (data summary + collected insights)
    and decides the next specific question to investigate.
    """
    llm = _llm()

    existing_insights = "\n".join(state["insights"]) if state["insights"] else "None yet."
    dqr = state.get("data_quality_report", "")
    dqr_section = f"\nData quality issues found by sanitizer (account for these):\n{dqr}\n" if dqr else ""

    messages = [
        SystemMessage(content=(
            "You are a data analyst planning the next step of an exploratory analysis.\n"
            "Output ONE specific, concrete analysis task to perform next.\n"
            "Rules:\n"
            "- Name the exact columns and metric to compute.\n"
            "- Do NOT repeat or rephrase an insight already found — pick something genuinely new.\n"
            "- Vary the angle: if you've done totals, try distributions or correlations next.\n"
            "- If data quality issues are listed below, plan tasks that work around them "
            "(e.g. filter outliers before aggregating, normalize rates before binning).\n"
            "- Do NOT write code — describe the task in 1-2 sentences only."
        )),
        HumanMessage(content=(
            f"Dataset info:\n{state['data_summary']}\n"
            f"{dqr_section}\n"
            f"Insights already found (do NOT repeat these):\n{existing_insights}\n\n"
            f"What NEW angle should we investigate? (iteration {state['iteration'] + 1})"
        )),
    ]

    response = llm.invoke(messages)
    plan = response.content.strip()
    print(f"\n[PLANNER] → {plan}")
    return {
        "current_plan": plan,
        "error_count": 0,
        "last_error": "",
        "token_usage": _add_tokens(state["token_usage"], response),
    }


# ── Node 2: Code Generator ────────────────────────────────────────────────────

def code_gen_node(state: AgentState) -> dict:
    """
    Takes the planner's task description and writes executable Python code.
    The code has access to `df` (the loaded DataFrame) and `pd` (pandas).
    """
    llm = _llm()

    dqr = state.get("data_quality_report", "")
    dqr_section = f"\nKnown data quality issues (sanitizer report):\n{dqr}\n" if dqr else ""

    messages = [
        SystemMessage(content=(
            "You are a Python data analyst. Write clean code to complete the task.\n"
            "Available variables: `df` (DataFrame), `pd` (pandas), `px` (plotly.express), `go` (plotly.graph_objects).\n"
            "Rules:\n"
            "- Use print() for ALL text output — this is how results are captured.\n"
            "- If the task involves a distribution, comparison, trend, or ranking across multiple values, create a Plotly chart.\n"
            "- Do NOT create a chart for a single scalar value (e.g. one correlation number, one mean).\n"
            "- For charts: assign the figure to a variable named exactly `fig` (e.g. fig = px.bar(...)).\n"
            "- Do NOT call fig.show() — just assign to `fig`.\n"
            "- Write ONLY the code block, no markdown fences, no explanation.\n"
            "- Always print key numeric results even when you also create a chart.\n"
            "- If data quality issues are listed below, your code MUST handle them: "
            "coerce types, filter outliers before charting, normalize rates, "
            "strip whitespace from categoricals."
        )),
        HumanMessage(content=(
            f"Dataset columns and types:\n{state['data_summary']}\n"
            f"{dqr_section}\n"
            f"Task: {state['current_plan']}"
        )),
    ]

    response = llm.invoke(messages)
    code = response.content.strip()
    code = code.removeprefix("```python").removeprefix("```").removesuffix("```").strip()

    print(f"\n[CODE GEN]\n{code}")
    return {
        "current_code": code,
        "token_usage": _add_tokens(state["token_usage"], response),
    }


# ── Node 3: Executor ──────────────────────────────────────────────────────────

def executor_node(state: AgentState) -> dict:
    """
    Runs generated code, captures stdout, and extracts any Plotly figure.
    run_python_code now returns a CodeResult object with .output and .chart_json.
    """
    result = run_python_code(state["current_code"])
    print(f"\n[EXECUTOR]\n{result.output}")
    if result.chart_json:
        print("[EXECUTOR] Chart generated.")

    if result.startswith("ERROR:"):
        return {
            "execution_result": result.output,
            "last_error": result.output,
            "charts": [],
        }

    new_charts = [result.chart_json] if result.chart_json else []
    return {
        "execution_result": result.output,
        "last_error": "",
        "charts": new_charts,   # appended to state["charts"] via Annotated operator.add
    }


# ── Node 3b: Error Fixer ──────────────────────────────────────────────────────

def error_fixer_node(state: AgentState) -> dict:
    """
    Phase 2 addition. Called when executor produces an error.

    Gets the broken code + traceback and asks the LLM to rewrite the code
    so it avoids the specific error. Returns the fixed code so executor
    can try again.

    Why pass the full traceback?
      The LLM needs the exact error type and line to fix it correctly.
      "TypeError: unexpected keyword argument 'raw'" is much more useful
      than "there was an error".
    """
    llm = _llm()
    attempt = state["error_count"] + 1
    print(f"\n[ERROR FIXER] Attempt {attempt}/3 — fixing error...")

    messages = [
        SystemMessage(content=(
            "You are a Python debugging expert. A pandas code snippet failed.\n"
            "Your job: rewrite the code to fix the error.\n"
            "Rules:\n"
            "- Available variables: `df` (DataFrame), `pd` (pandas), `px` (plotly.express), `go` (plotly.graph_objects).\n"
            "- Use print() for ALL text output.\n"
            "- If the original code created a chart, keep the chart in the fixed version. Assign it to `fig`.\n"
            "- Do NOT call fig.show() — just assign to `fig`.\n"
            "- Write ONLY the fixed code, no markdown fences, no explanation.\n"
            "- Keep the same analysis goal, just fix the bug."
        )),
        HumanMessage(content=(
            f"Original task: {state['current_plan']}\n\n"
            f"Broken code:\n{state['current_code']}\n\n"
            f"Error traceback:\n{state['last_error']}\n\n"
            "Rewrite the code to fix this error."
        )),
    ]

    response = llm.invoke(messages)
    fixed_code = response.content.strip()
    fixed_code = fixed_code.removeprefix("```python").removeprefix("```").removesuffix("```").strip()

    print(f"\n[ERROR FIXER] Fixed code:\n{fixed_code}")
    return {
        "current_code": fixed_code,
        "error_count": state["error_count"] + 1,
        "token_usage": _add_tokens(state["token_usage"], response),
    }


# ── Routing: after executor ───────────────────────────────────────────────────

def after_executor(state: AgentState) -> str:
    """
    Conditional edge called after every executor run.

    Decision tree:
      - No error → go to reflector (normal path)
      - Error + retries remaining → go to error_fixer
      - Error + retries exhausted → go to reflector anyway (log it and move on)

    MAX_RETRIES = 3: chosen to balance quality vs. Groq rate limits.
    """
    MAX_RETRIES = 3
    is_error = state["execution_result"].startswith("ERROR:")

    if not is_error:
        if state.get("phase") == "sanitizing":
            return "sanitizer_store"
        return "reflector"

    if state["error_count"] < MAX_RETRIES:
        return "error_fixer"

    # Exhausted retries — store or reflect depending on phase
    print(f"\n[ROUTER] Max retries reached for this task. Moving on.")
    if state.get("phase") == "sanitizing":
        return "sanitizer_store"
    return "reflector"


# ── Node 4: Reflector ─────────────────────────────────────────────────────────

def reflector_node(state: AgentState) -> dict:
    """
    Reads the execution output and decides:
    1. Is this result a useful insight? Extract it.
    2. Have we found enough insights? Set iteration count.

    Does NOT route — routing is handled by the conditional edge below.
    """
    llm = _llm()

    messages = [
        SystemMessage(content=(
            "You are reviewing the output of a data analysis step.\n"
            "Extract the key insight from the output in 1-2 clear sentences.\n"
            "Start your response with 'INSIGHT:' followed by the finding.\n"
            "If the output is an error or empty, start with 'ERROR:' and describe what went wrong."
        )),
        HumanMessage(content=(
            f"Task that was run: {state['current_plan']}\n\n"
            f"Output:\n{state['execution_result']}"
        )),
    ]

    response = llm.invoke(messages)
    insight = response.content.strip()
    print(f"\n[REFLECTOR] → {insight}")

    return {
        "insights": [insight],
        "iteration": state["iteration"] + 1,
        "token_usage": _add_tokens(state["token_usage"], response),
    }


# ── Node 5: Summarizer ────────────────────────────────────────────────────────

def summarizer_node(state: AgentState) -> dict:
    """
    Called once when the loop ends. Combines all insights into a
    structured executive summary.
    """
    llm = _llm()

    insights_text = "\n".join(state["insights"])

    dqr = state.get("data_quality_report", "")
    dqr_section = (
        f"\nData quality caveats identified before analysis:\n{dqr}\n"
        if dqr else ""
    )

    messages = [
        SystemMessage(content=(
            "You are a senior data analyst writing an executive summary.\n"
            "Synthesize the findings into a clear, structured report.\n"
            "Use sections: Key Findings, Notable Patterns, Recommendations.\n"
            "If data quality caveats are provided, add a brief 'Data Quality Caveats' "
            "section at the end noting which findings may be affected.\n"
            "Be concise — 200-300 words max."
        )),
        HumanMessage(content=(
            f"Dataset: {state['file_path']}\n"
            f"{dqr_section}\n"
            f"Analysis findings:\n{insights_text}"
        )),
    ]

    response = llm.invoke(messages)
    summary = response.content.strip()
    print(f"\n{'='*60}\n[FINAL SUMMARY]\n{summary}\n{'='*60}")
    return {
        "final_summary": summary,
        "token_usage": _add_tokens(state["token_usage"], response),
    }


# ── Routing logic ─────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """
    Conditional edge function called after the reflector.
    Returns the name of the next node to route to.

    Why a function instead of a fixed edge?
      This is where the 'agentic' part lives — the graph decides
      its own next step based on current state, not a hardcoded sequence.
    """
    if state["iteration"] >= state["max_iterations"]:
        return "summarizer"

    # If the last insight was an error, still continue (Phase 2 will handle retries)
    return "planner"


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("sanitizer", sanitizer_node)
    graph.add_node("sanitizer_store", sanitizer_store_node)
    graph.add_node("planner", planner_node)
    graph.add_node("code_gen", code_gen_node)
    graph.add_node("executor", executor_node)
    graph.add_node("error_fixer", error_fixer_node)
    graph.add_node("reflector", reflector_node)
    graph.add_node("summarizer", summarizer_node)

    # Fixed edges
    graph.set_entry_point("sanitizer")
    graph.add_edge("sanitizer", "code_gen")      # task-setter goes straight to code_gen
    graph.add_edge("sanitizer_store", "planner") # DQ report stored → begin analysis
    graph.add_edge("planner", "code_gen")
    graph.add_edge("code_gen", "executor")
    graph.add_edge("error_fixer", "executor")    # fixer always goes back to executor
    graph.add_edge("summarizer", END)

    # executor routes conditionally (phase-aware)
    graph.add_conditional_edges(
        "executor",
        after_executor,
        {
            "sanitizer_store": "sanitizer_store",
            "reflector": "reflector",
            "error_fixer": "error_fixer",
        },
    )

    # Reflector decides whether to loop or finish
    graph.add_conditional_edges(
        "reflector",
        should_continue,
        {
            "planner": "planner",
            "summarizer": "summarizer",
        },
    )

    return graph.compile()


def _build_initial_state(file_path: str, max_iterations: int) -> AgentState:
    data_summary = load_csv(file_path)
    return {
        "file_path": file_path,
        "data_summary": data_summary,
        "current_plan": "",
        "current_code": "",
        "execution_result": "",
        "insights": [],
        "charts": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "final_summary": "",
        "error_count": 0,
        "last_error": "",
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "data_quality_report": "",
        "phase": "sanitizing",
    }


def stream_analysis(
    file_path: str,
    max_iterations: int = 5,
    provider: str = "groq",
    api_key: str | None = None,
    model: str | None = None,
):
    """
    Generator used by Streamlit for live updates.

    graph.stream() yields {node_name: state_snapshot} after every node fires.
    Streamlit calls next() on this generator inside a loop and updates the UI
    incrementally — the user sees each step as it happens instead of waiting
    for the full run to complete.

    Yields dicts like:
      {"node": "planner",  "state": {...}}
      {"node": "code_gen", "state": {...}}
      ...
    """
    set_llm_config(provider, api_key, model)

    initial_state = _build_initial_state(file_path, max_iterations)
    graph = build_graph()

    for chunk in graph.stream(initial_state):
        # chunk = {node_name: full_state_after_that_node}
        node_name = next(iter(chunk))
        state = chunk[node_name]
        yield {"node": node_name, "state": state}


def run_analysis(
    file_path: str,
    max_iterations: int = 5,
    provider: str = "groq",
    api_key: str | None = None,
    model: str | None = None,
) -> dict:
    """CLI entry point — blocks until complete, returns final state."""
    set_llm_config(provider, api_key, model)
    initial_state = _build_initial_state(file_path, max_iterations)
    print(f"[INIT] {initial_state['data_summary']}\n")
    graph = build_graph()
    return graph.invoke(initial_state)
