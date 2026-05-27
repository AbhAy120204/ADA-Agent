"""
Agent tools — functions the LLM can call by name.

Why this design:
  The LLM never imports pandas directly. It writes code as a string,
  and we run it here. This gives us one place to add sandboxing later
  (swap exec() for E2B) without touching any graph logic.
"""

import io
import re
import sys
import traceback
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Force Plotly to never open a browser window.
# "json" renderer means fig.show() writes JSON to stdout instead of
# launching a browser tab — and we strip that from output below anyway.
pio.renderers.default = "json"


# Global store: once a CSV is loaded, every code execution can use `df`
_dataframe_store: dict[str, pd.DataFrame] = {}


def load_csv(file_path: str) -> str:
    """
    Load a CSV from disk into memory as a pandas DataFrame.
    Returns a plain-text summary so the LLM knows what it loaded.
    """
    try:
        df = pd.read_csv(file_path)
        _dataframe_store["df"] = df

        summary = (
            f"CSV loaded successfully.\n"
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            f"Columns: {', '.join(df.columns.tolist())}\n"
            f"Dtypes:\n{df.dtypes.to_string()}\n"
            f"First 3 rows:\n{df.head(3).to_string()}"
        )
        return summary
    except Exception as e:
        return f"ERROR loading CSV: {e}"


class CodeResult:
    """
    Return type for run_python_code.
    Carries both the text output and an optional Plotly figure JSON.

    Why a class instead of a tuple?
      graph.py checks result.startswith("ERROR:") in several places.
      Keeping output as a plain string field means those checks stay unchanged.
      The chart is bonus data — present when the code created a `fig` variable,
      absent otherwise.
    """
    def __init__(self, output: str, chart_json: str | None = None):
        self.output = output
        self.chart_json = chart_json  # serialized plotly figure, or None

    def startswith(self, prefix: str) -> bool:
        return self.output.startswith(prefix)

    def __str__(self) -> str:
        return self.output


def _sanitize_code(code: str) -> str:
    """
    Pre-process code before execution to prevent side effects.

    - Remove fig.show() calls: LLMs frequently write these despite instructions.
      With pio.renderers.default="json" they'd dump JSON to stdout; stripping
      them entirely is cleaner.
    - Remove bare import statements for plotly — already injected in namespace.
    """
    # Remove any fig.show(...) or fig.show() calls
    code = re.sub(r"\bfig\.show\s*\([^)]*\)", "", code)
    # Remove standalone plotly import lines (px/go already injected)
    code = re.sub(r"^\s*import plotly.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"^\s*from plotly.*$", "", code, flags=re.MULTILINE)
    return code


def _clean_output(raw: str) -> str:
    """
    Remove Plotly Figure repr from captured stdout.

    When code does print(fig), Plotly dumps a multi-line Figure({...}) repr.
    This is noisy, misleading for the reflector LLM, and not actual analysis output.
    We detect it by the Figure({ prefix and strip those lines.
    """
    lines = raw.splitlines()
    cleaned = []
    skip = False
    for line in lines:
        if line.strip().startswith("Figure({"):
            skip = True
        if skip:
            # Figure repr ends when we hit a standalone closing paren at indent 0
            if line.strip() == "})":
                skip = False
            continue
        cleaned.append(line)
    result = "\n".join(cleaned).strip()
    return result if result else "Code ran successfully (no text output)."


def run_python_code(code: str) -> "CodeResult":
    """
    Execute a Python code string, capture stdout, and extract any Plotly figure.

    Available in the execution namespace:
      df  — the loaded DataFrame
      pd  — pandas
      px  — plotly.express  (e.g. px.bar, px.line, px.scatter)
      go  — plotly.graph_objects  (e.g. go.Figure, go.Bar)

    Chart convention: if the code assigns a Plotly figure to a variable
    named `fig`, we serialize it to JSON alongside the text output.
    """
    if "df" not in _dataframe_store:
        return CodeResult("ERROR: No CSV loaded. Call load_csv first.")

    code = _sanitize_code(code)

    stdout_capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_capture

    exec_globals = {
        "df": _dataframe_store["df"].copy(),
        "pd": pd,
        "px": px,
        "go": go,
    }

    try:
        exec(code, exec_globals)  # noqa: S102
        raw_output = stdout_capture.getvalue()
        text = _clean_output(raw_output)

        chart_json = None
        fig = exec_globals.get("fig")
        if fig is not None and hasattr(fig, "to_json"):
            chart_json = fig.to_json()

        return CodeResult(text, chart_json)
    except Exception:
        error = traceback.format_exc()
        return CodeResult(f"ERROR:\n{error}")
    finally:
        sys.stdout = old_stdout


def get_dataframe_info() -> str:
    """
    Return column names, dtypes, and null counts — useful for the planner
    to understand the data before writing analysis code.
    """
    if "df" not in _dataframe_store:
        return "No DataFrame loaded yet."

    df = _dataframe_store["df"]
    null_info = df.isnull().sum()
    null_str = null_info[null_info > 0].to_string() if null_info.any() else "No nulls found."

    return (
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        f"Columns & types:\n{df.dtypes.to_string()}\n"
        f"Null counts:\n{null_str}\n"
        f"Numeric summary:\n{df.describe().to_string()}"
    )
