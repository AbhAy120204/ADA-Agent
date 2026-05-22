"""
Agent tools — functions the LLM can call by name.

Why this design:
  The LLM never imports pandas directly. It writes code as a string,
  and we run it here. This gives us one place to add sandboxing later
  (swap exec() for E2B) without touching any graph logic.
"""

import io
import sys
import traceback
import pandas as pd


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


def run_python_code(code: str) -> str:
    """
    Execute a Python code string and return stdout + any errors.

    The variable `df` (the loaded DataFrame) is available inside the code.
    We capture print() output and return it as a string.

    Why exec() here and not a subprocess?
      For Phase 1 simplicity. Phase 2 will add a subprocess sandbox
      so user-written code can't affect the main process.
    """
    if "df" not in _dataframe_store:
        return "ERROR: No CSV loaded. Call load_csv first."

    # Capture stdout so print() calls inside the code come back to us
    stdout_capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_capture

    # Inject `df` and `pd` into the execution namespace
    exec_globals = {
        "df": _dataframe_store["df"].copy(),
        "pd": pd,
    }

    try:
        exec(code, exec_globals)  # noqa: S102
        output = stdout_capture.getvalue()
        return output if output else "Code ran successfully (no output printed)."
    except Exception:
        error = traceback.format_exc()
        return f"ERROR:\n{error}"
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
