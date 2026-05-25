"""
The ReAct loop implemented as a LangGraph StateGraph.

Node flow (Phase 2):
  planner → code_gen → executor ──success──→ reflector → (planner OR summarizer)
                            └──error──→ error_fixer → executor (retry, max 3x)
                                              └──3rd failure──→ reflector anyway

Why a dedicated error_fixer node instead of retrying inside executor?
  Separation of concerns: executor just runs code and reports.
  error_fixer is the only place that knows about retries — it holds the
  error_count state and the traceback. Clean to test, clean to extend.
"""

from typing import TypedDict, Annotated
import operator

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from agent.tools import load_csv, run_python_code, get_dataframe_info


# ── State definition ──────────────────────────────────────────────────────────
# TypedDict tells LangGraph exactly what fields live in the state.
# Annotated[list, operator.add] means "append to this list" instead of replacing it.

class AgentState(TypedDict):
    file_path: str                              # CSV path provided by user
    data_summary: str                           # Output of load_csv()
    current_plan: str                           # What the planner decided to explore next
    current_code: str                           # Code written by code_gen
    execution_result: str                       # Output of running the code
    insights: Annotated[list[str], operator.add]  # Accumulated findings (appends each loop)
    iteration: int                              # How many ReAct loops we've done
    max_iterations: int                         # Safety limit to prevent infinite loops
    final_summary: str                          # Executive summary produced at the end
    # Phase 2 additions
    error_count: int                            # Retry attempts for the current task (resets each plan)
    last_error: str                             # Traceback from the last failed execution


# ── LLM setup ─────────────────────────────────────────────────────────────────

def _get_llm() -> ChatGroq:
    """
    Returns a Groq LLM. We use llama-3.1-8b-instant because:
    - It's fast (low latency for iterative loops)
    - Free tier on Groq
    - Good enough for code generation and analysis reasoning
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,  # 0 = deterministic, important for code generation
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ── Node 1: Planner ───────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> dict:
    """
    Looks at what we know so far (data summary + collected insights)
    and decides the next specific question to investigate.
    """
    llm = _get_llm()

    existing_insights = "\n".join(state["insights"]) if state["insights"] else "None yet."

    messages = [
        SystemMessage(content=(
            "You are a data analyst planning the next step of an exploratory analysis.\n"
            "You will be given a summary of the dataset and insights already found.\n"
            "Output ONE specific, concrete analysis task to perform next.\n"
            "Be specific: name the exact columns and what to compute.\n"
            "Do NOT write code — just describe what to do in 1-2 sentences."
        )),
        HumanMessage(content=(
            f"Dataset info:\n{state['data_summary']}\n\n"
            f"Insights already found:\n{existing_insights}\n\n"
            f"What should we analyze next? (iteration {state['iteration'] + 1})"
        )),
    ]

    response = llm.invoke(messages)
    plan = response.content.strip()
    print(f"\n[PLANNER] → {plan}")
    # Reset error counter — each new plan gets 3 fresh retry attempts
    return {"current_plan": plan, "error_count": 0, "last_error": ""}


# ── Node 2: Code Generator ────────────────────────────────────────────────────

def code_gen_node(state: AgentState) -> dict:
    """
    Takes the planner's task description and writes executable Python code.
    The code has access to `df` (the loaded DataFrame) and `pd` (pandas).
    """
    llm = _get_llm()

    messages = [
        SystemMessage(content=(
            "You are a Python data analyst. Write clean pandas code to complete the task.\n"
            "Rules:\n"
            "- The variable `df` contains the DataFrame. `pd` is already imported.\n"
            "- Use print() for ALL output — this is how results are captured.\n"
            "- Write ONLY the code block, no markdown fences, no explanation.\n"
            "- Keep it short and focused on the task."
        )),
        HumanMessage(content=(
            f"Dataset columns and types:\n{state['data_summary']}\n\n"
            f"Task: {state['current_plan']}"
        )),
    ]

    response = llm.invoke(messages)
    # Strip any accidental markdown code fences the LLM might add
    code = response.content.strip()
    code = code.removeprefix("```python").removeprefix("```").removesuffix("```").strip()

    print(f"\n[CODE GEN]\n{code}")
    return {"current_code": code}


# ── Node 3: Executor ──────────────────────────────────────────────────────────

def executor_node(state: AgentState) -> dict:
    """
    Runs the generated code and captures the output.
    Now stores the raw error in last_error so error_fixer can read it.
    """
    result = run_python_code(state["current_code"])
    print(f"\n[EXECUTOR]\n{result}")

    if result.startswith("ERROR:"):
        return {"execution_result": result, "last_error": result}
    return {"execution_result": result, "last_error": ""}


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
    llm = _get_llm()
    attempt = state["error_count"] + 1
    print(f"\n[ERROR FIXER] Attempt {attempt}/3 — fixing error...")

    messages = [
        SystemMessage(content=(
            "You are a Python debugging expert. A pandas code snippet failed.\n"
            "Your job: rewrite the code to fix the error.\n"
            "Rules:\n"
            "- The variable `df` contains the DataFrame. `pd` is already imported.\n"
            "- Use print() for ALL output.\n"
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
        return "reflector"

    if state["error_count"] < MAX_RETRIES:
        return "error_fixer"

    # Exhausted retries — send to reflector which will log this as an ERROR insight
    print(f"\n[ROUTER] Max retries reached for this task. Moving on.")
    return "reflector"


# ── Node 4: Reflector ─────────────────────────────────────────────────────────

def reflector_node(state: AgentState) -> dict:
    """
    Reads the execution output and decides:
    1. Is this result a useful insight? Extract it.
    2. Have we found enough insights? Set iteration count.

    Does NOT route — routing is handled by the conditional edge below.
    """
    llm = _get_llm()

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
        "insights": [insight],               # appended to the list (see Annotated above)
        "iteration": state["iteration"] + 1,
    }


# ── Node 5: Summarizer ────────────────────────────────────────────────────────

def summarizer_node(state: AgentState) -> dict:
    """
    Called once when the loop ends. Combines all insights into a
    structured executive summary.
    """
    llm = _get_llm()

    insights_text = "\n".join(state["insights"])

    messages = [
        SystemMessage(content=(
            "You are a senior data analyst writing an executive summary.\n"
            "Synthesize the findings into a clear, structured report.\n"
            "Use sections: Key Findings, Notable Patterns, Recommendations.\n"
            "Be concise — 200-300 words max."
        )),
        HumanMessage(content=(
            f"Dataset: {state['file_path']}\n\n"
            f"Analysis findings:\n{insights_text}"
        )),
    ]

    response = llm.invoke(messages)
    summary = response.content.strip()
    print(f"\n{'='*60}\n[FINAL SUMMARY]\n{summary}\n{'='*60}")
    return {"final_summary": summary}


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

    # Register all nodes (error_fixer is new in Phase 2)
    graph.add_node("planner", planner_node)
    graph.add_node("code_gen", code_gen_node)
    graph.add_node("executor", executor_node)
    graph.add_node("error_fixer", error_fixer_node)
    graph.add_node("reflector", reflector_node)
    graph.add_node("summarizer", summarizer_node)

    # Fixed edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "code_gen")
    graph.add_edge("code_gen", "executor")
    graph.add_edge("error_fixer", "executor")  # fixer always goes back to executor
    graph.add_edge("summarizer", END)

    # Phase 2: executor now routes conditionally (was a fixed edge in Phase 1)
    graph.add_conditional_edges(
        "executor",
        after_executor,
        {
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


def run_analysis(file_path: str, max_iterations: int = 5) -> dict:
    """
    Entry point for running the full analysis pipeline.
    Returns the final state with all insights and the summary.
    """
    # Step 1: load the CSV so tools.py has it in memory
    data_summary = load_csv(file_path)
    print(f"[INIT] {data_summary}\n")

    initial_state: AgentState = {
        "file_path": file_path,
        "data_summary": data_summary,
        "current_plan": "",
        "current_code": "",
        "execution_result": "",
        "insights": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "final_summary": "",
        "error_count": 0,
        "last_error": "",
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)
    return final_state
