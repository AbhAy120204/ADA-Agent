<div align="center">

# 🔍 Autonomous Data Analyst Agent

### Give it a CSV. It thinks, writes code, fixes its own bugs, and hands you insights.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://autonomous-data-analyst-agent-hun39rzqecosyvx5pqu3rg.streamlit.app/)
[![LangGraph](https://img.shields.io/badge/LangGraph-ReAct%20Loop-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-llama--3.1--8b-F55036?style=for-the-badge&logo=lightning&logoColor=white)](https://console.groq.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

</div>

---

## What makes this different

Most "AI data analysis" tools are wrappers — you ask a question, it writes SQL, done. This agent is different in one key way: **it loops**.

```
Upload CSV
    │
    ▼
🧠 PLAN  ──→  ✍️ WRITE CODE  ──→  ⚙️ RUN IT
                                       │
                              ┌────────┴────────┐
                           success            error
                              │                 │
                              ▼                 ▼
                         🪞 REFLECT        🔧 FIX IT  ──→  ⚙️ RUN AGAIN
                              │                              (max 3 retries)
                         ┌────┴────┐
                    more to find   done
                         │         │
                         ▼         ▼
                      🧠 PLAN   📋 SUMMARY
```

The agent decides what to investigate next based on what it already found. It writes its own pandas code, runs it, reads the error if it fails, rewrites the code, and tries again — all without you touching anything.

---

## Live demo

**👉 [Try it here](https://autonomous-data-analyst-agent-hun39rzqecosyvx5pqu3rg.streamlit.app/)**

Bring your own [free Groq API key](https://console.groq.com) (takes 2 minutes to get). Upload any CSV.

---

## The ReAct loop in action

The UI shows the agent's internal reasoning in real time — collapsed by default, expandable like ChatGPT's "Thinking" panel:

| Step | What you see |
|------|-------------|
| 🧠 **Plan** | "Analyze revenue by sales rep, group by region" |
| ✍️ **Code** | `df.groupby('sales_rep')['revenue'].sum()` |
| ⚙️ **Execute** | `Carol: $91,748 · Alice: $88,398 · Bob: $64,348` |
| 🔧 **Error fix** | Auto-rewrites broken code from traceback (up to 3x) |
| 🪞 **Insight** | "Carol is the top performer with $91k revenue" |
| 📋 **Summary** | Full executive report with Key Findings + Recommendations |

---

## Why this is hard to build (and why it matters)

| Naive approach | This agent |
|---|---|
| Single LLM call → answer | Iterative loop — each finding informs the next question |
| Crashes on bad code | Reads its own traceback, rewrites, retries up to 3x |
| Fixed pipeline (step 1 → step 2 → step 3) | Dynamic graph — agent decides its own next step |
| Shows final answer only | Streams every node live — you watch it think |
| One model hardcoded | BYOK: bring your own Groq/Gemini key from the sidebar |

---

## Tech stack

```
Agent orchestration  →  LangGraph (StateGraph with conditional edges)
LLM inference        →  Groq API  (llama-3.1-8b-instant, free tier)
Code execution       →  Python exec() in isolated namespace with stdout capture
Data manipulation    →  Pandas
UI + deployment      →  Streamlit Community Cloud
```

**Why LangGraph over a plain `while` loop?**
LangGraph gives you stateful nodes, conditional routing, and checkpointing. The error-recovery branch (`executor → error_fixer → executor`) is a single conditional edge — clean, testable, extendable. A plain loop would need nested `try/except` and manual state threading.

**Why Groq over OpenAI?**
Free tier, fast inference (~500 tok/s on Llama 3.1), no credit card needed. The agent makes 4–8 LLM calls per run — latency matters in a loop.

---

## Run locally

```bash
git clone https://github.com/AbhAy120204/-Autonomous-Data-Analyst-Agent.git
cd -Autonomous-Data-Analyst-Agent

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Add your Groq key
echo "GROQ_API_KEY=gsk_..." > .env

# Launch UI
streamlit run app.py

# Or CLI
python main.py --file data/examples/sales.csv --iterations 5
```

---

## Project structure

```
├── agent/
│   ├── graph.py      # LangGraph StateGraph — the ReAct loop
│   └── tools.py      # load_csv, run_python_code, get_dataframe_info
├── app.py            # Streamlit UI (streaming, collapsible thinking panel)
├── main.py           # CLI entry point
└── data/examples/    # Sample datasets
```

---

## Roadmap

- [x] Phase 1 — Core ReAct loop (plan → code → execute → reflect)
- [x] Phase 2 — Error recovery (auto-fix broken code, 3 retries)
- [x] Phase 3 — Streamlit UI with live streaming
- [ ] Phase 4 — Plotly chart generation
- [ ] Phase 5 — SQL / database support
- [ ] Phase 6 — Eval framework + LLM observability
- [ ] Phase 7 — Multi-model BYOK (Groq / Gemini / Ollama)

---

## Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free, no credit card)
3. Create an API key
4. Paste it in the sidebar → Run analysis

---

<div align="center">
Built by <a href="https://github.com/AbhAy120204">Abhay Tiwari</a> · LangGraph + Groq + Streamlit
</div>
