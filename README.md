# Autonomous Data Analyst Agent — Project Plan

## What We're Building

An AI agent that takes a CSV file or database connection, then autonomously:
1. Explores the data (shape, types, nulls, distributions)
2. Writes Python/SQL code to answer questions about it
3. Executes that code in a safe sandbox
4. Recovers from errors by reading the traceback and fixing the code
5. Generates Plotly charts
6. Produces an executive summary

The key word is **autonomous** — the agent loops (Plan → Code → Execute → Reflect → Repeat)
until it has enough insights, without human hand-holding between steps. This is a true ReAct
(Reasoning + Acting) loop, not a fixed pipeline.

---

## Why This Stands Out (vs Open Source Alternatives)

After researching the landscape (Vanna, EDAgent, OpenChatBI, Reasonlytics, Quelmap):

| What exists | What's missing |
|---|---|
| SQL-only agents (Vanna) | Combined pandas + SQL + ReAct loop |
| Fixed 8-step pipelines (EDAgent) | Dynamic, hypothesis-driven exploration |
| Local-only tools | Deployed, shareable demo |
| No cost transparency | Token/cost meter per session |
| No free-tier LLM support | Groq (free) + Gemini (free) + BYOK |

---

## Tech Stack — Why Each Piece

### LangGraph
- Chosen over simple LangChain chains because it gives us a **stateful graph** (nodes + edges).
- We can model the ReAct loop as: `plan → code → execute → reflect → (loop or end)`.
- Has built-in checkpointing so we can resume failed analyses.
- Alternative considered: CrewAI (multi-agent but harder to control loop logic).

### Groq (Free LLM API)
- Groq runs Llama 3.1/Mixtral at **very fast inference, free tier**.
- Google Gemini is the fallback free option.
- Ollama is the fully local option (no API key needed).
- All three will be supported. User picks in the sidebar.

### Streamlit
- Pure Python UI — no React, no FastAPI, no frontend/backend split.
- One file can be both the UI and the logic.
- Free deployment via Streamlit Community Cloud.
- Alternative considered: Gradio (less flexible for custom layouts).

### Python subprocess sandbox (Week 1-2) → E2B (future)
- For the learning-focused incremental build, we start with `subprocess` + `restricted exec`.
- E2B (production sandbox) is introduced later once the agent loop works.
- This way we understand WHY a sandbox is needed before we use a managed one.

### Plotly
- Generates interactive charts that embed directly in Streamlit.
- Agent outputs chart spec as JSON, Streamlit renders it.
- Alternative considered: Matplotlib (static, less impressive for portfolio).

### SQLAlchemy
- Unified interface to talk to SQLite, PostgreSQL, MySQL from the same agent code.
- Agent doesn't need to know which database it's talking to.

---

## Architecture: The ReAct Loop

```
User uploads CSV / connects DB
         │
         ▼
  ┌─────────────┐
  │   PLANNER   │  ← LLM decides: what to explore next?
  └──────┬──────┘
         │ plan
         ▼
  ┌─────────────┐
  │  CODE GEN   │  ← LLM writes Python/SQL to answer the plan
  └──────┬──────┘
         │ code
         ▼
  ┌─────────────┐
  │   EXECUTE   │  ← Run code in sandbox, capture output + errors
  └──────┬──────┘
         │ result / error
         ▼
  ┌─────────────┐
  │   REFLECT   │  ← LLM reads output, decides: done? retry? next question?
  └──────┬──────┘
         │
    ┌────┴────┐
    │         │
  loop      end
    │         │
    ▲         ▼
    └──────  ┌──────────────┐
             │  SUMMARIZER  │  ← Final executive summary + charts
             └──────────────┘
```

---

## Incremental Development Phases

> **Rule**: Each phase produces a working, runnable thing. No phase is "just setup".

---

### Phase 1 — Bare Minimum Working Agent (Week 1)

**Goal**: A terminal script where you pass a CSV path and the agent prints insights.

**What we build:**
- `agent/tools.py` — two tools: `run_python_code(code)` and `load_csv(path)`
- `agent/graph.py` — LangGraph graph with 4 nodes: planner, code_gen, executor, reflector
- `main.py` — CLI entry point: `python main.py --file data.csv`
- `requirements.txt`

**What you learn in this phase:**
- How LangGraph StateGraph works (nodes, edges, state dict)
- How to define tools the LLM can call
- What a ReAct loop looks like in code (not theory)

**Success check**: `python main.py --file titanic.csv` prints 5+ insights about the data.

**GitHub push**: `feat: working CLI agent with ReAct loop`

---

### Phase 2 — Error Recovery Loop (Week 2)

**Goal**: Agent doesn't crash when its code fails — it reads the error and fixes it.

**What we add:**
- `agent/nodes/executor.py` — captures stdout, stderr, tracebacks separately
- Retry logic in the graph: executor → reflect → code_gen (on error, max 3 retries)
- State tracks `error_count` and `last_error`
- LLM prompt for "fix this code given this error"

**What you learn:**
- Why stateful graphs beat simple chains (you need to pass error context across nodes)
- How to design prompts for error recovery
- How `try/except` + `subprocess` works for safe code execution

**Success check**: Deliberately break a CSV (missing column) — agent catches and corrects.

**GitHub push**: `feat: error recovery with retry loop`

---

### Phase 3 — Streamlit UI (Week 3)

**Goal**: Everything Phase 1+2 does, but in a browser with a drag-and-drop file upload.

**What we add:**
- `app.py` — Streamlit app
- File uploader widget → saves to temp path → triggers agent
- `st.expander` for each agent step (shows plan, code, output)
- Agent thoughts stream to UI in real-time using `st.empty()` + generator pattern
- LLM selector in sidebar (Groq / Gemini / Ollama)

**What you learn:**
- How Streamlit session state works (keeping agent output across reruns)
- How to stream LLM output to Streamlit (generator + `st.write_stream`)
- Why UI feedback matters for agentic systems (user needs to see the loop)

**Success check**: Upload CSV in browser → watch agent think step by step → see insights.

**GitHub push**: `feat: streamlit UI with streaming agent thoughts`

---

### Phase 4 — Visualizations (Week 4)

**Goal**: Agent generates and displays Plotly charts alongside text insights.

**What we add:**
- `agent/tools.py` — new tool: `generate_chart(df, chart_type, x, y, title)`
- Agent prompted to call chart tool when it has a visualization-worthy finding
- Charts serialized as JSON in agent state → rendered by Streamlit via `st.plotly_chart`
- Agent picks chart type based on data (bar for categorical, line for time-series, etc.)

**What you learn:**
- How to design tools that return structured data (not just text)
- How to pass complex objects (chart JSON) through LangGraph state
- Plotly's JSON schema for chart specs

**Success check**: Agent produces at least 2 charts per CSV automatically.

**GitHub push**: `feat: automatic chart generation`

---

### Phase 5 — SQL + Database Support (Week 5)

**Goal**: Connect a SQLite/PostgreSQL database instead of a CSV.

**What we add:**
- `agent/tools.py` — `run_sql_query(query)` tool using SQLAlchemy
- Schema introspection tool: `get_table_schema()` → agent knows column names/types
- Sidebar: user enters connection string or uploads SQLite file
- Agent decides tool to use (pandas vs SQL) based on data source type

**What you learn:**
- How SQLAlchemy abstracts different databases
- Why schema introspection is critical for text-to-SQL agents
- How to give an agent "awareness" of its environment through tools

**Success check**: Connect to a SQLite DB → agent writes and runs SQL → shows results.

**GitHub push**: `feat: SQL and database support`

---

### Phase 6 — Evaluation + Observability (Week 6)

**Goal**: Know whether the agent is actually good, with numbers to prove it.

**What we add:**
- `evals/` folder with 5 benchmark CSVs (Titanic, Iris, Sales, Weather, Finance)
- Each benchmark has a `ground_truth.json` with expected insights
- `evals/run_benchmarks.py` — runs agent on all 5, scores insight coverage
- LangSmith or Maxim tracing integration (free tier)
- Token usage counter displayed in Streamlit sidebar

**What you learn:**
- How to evaluate LLM agent output (not just "does it run" but "is it correct")
- What LLMOps observability means (tracing every node, every token)
- Why evaluation matters for portfolio projects (shows engineering rigor)

**GitHub push**: `feat: eval framework + observability`

---

### Phase 7 — BYOK + Multi-Model Support (Week 7)

**Goal**: Any user can bring their own API key and choose their model.

**What we add:**
- Streamlit sidebar: model selector (Groq / Gemini / Ollama / OpenAI)
- API key input field (stored in `st.session_state`, never persisted to disk)
- `agent/llm_factory.py` — returns the right LangChain LLM based on selection
- Fallback chain: if Groq fails → try Gemini → warn user

**What you learn:**
- How LangChain's LLM abstraction works (swap providers with one line)
- Why BYOK matters for deployed apps (you can't pay for everyone's queries)
- How to handle secrets safely in Streamlit

**GitHub push**: `feat: multi-model support with BYOK`

---

### Phase 8 — Deploy + Portfolio Polish (Week 8)

**Goal**: Live public URL. Resume-ready README. Demo video.

**What we add:**
- `requirements.txt` pinned versions
- `.streamlit/config.toml` for theme + deployment settings
- Deploy to Streamlit Community Cloud (free, connects to GitHub repo)
- README: architecture diagram, demo GIF, benchmark results, "How to run" section
- 3 example analyses (Titanic, Sales data, Iris) as screenshots in README

**What you learn:**
- How Streamlit Cloud deployment works (it just reads your repo)
- How to write a portfolio README that communicates impact

**GitHub push**: `feat: deployment config + v1.0 release tag`

---

## Repository Structure (Final Target)

```
autonomous-data-analyst-agent/
│
├── agent/
│   ├── __init__.py
│   ├── graph.py          # LangGraph state machine (the ReAct loop)
│   ├── nodes/
│   │   ├── planner.py    # Decides what question to answer next
│   │   ├── code_gen.py   # Writes Python/SQL code
│   │   ├── executor.py   # Runs code safely, captures output/errors
│   │   ├── reflector.py  # Reads output, decides next action
│   │   └── summarizer.py # Final executive summary
│   ├── tools.py          # Agent-callable tools (run_code, load_csv, chart, sql)
│   └── llm_factory.py    # Returns LLM based on user's model choice
│
├── evals/
│   ├── benchmarks/       # Test CSVs with known expected insights
│   ├── ground_truth/     # Expected outputs per benchmark
│   └── run_benchmarks.py # Evaluation script
│
├── data/
│   └── examples/         # Sample datasets for demo
│
├── app.py                # Streamlit UI (entry point for web)
├── main.py               # CLI entry point (entry point for terminal)
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── README.md
```

---

## Rules for This Build

1. **Explain before adding**: Before introducing any new library or pattern, we discuss why it's needed.
2. **Working at each phase**: Every phase ends with something you can run.
3. **Incremental GitHub pushes**: One focused commit per phase, clearly labeled.
4. **No premature abstraction**: Don't build `llm_factory.py` in Phase 1. Add it when needed.
5. **Learning notes**: Key concepts explained inline as comments in the code.

---

## Free API Resources

| Provider | Free Tier | How to Get Key |
|---|---|---|
| Groq | 14,400 req/day, fast Llama 3.1 | console.groq.com |
| Google Gemini | 15 req/min, Gemini 1.5 Flash | aistudio.google.com |
| Ollama | Unlimited (runs locally) | ollama.com |
| Streamlit Cloud | Free deploy for public repos | share.streamlit.io |
| LangSmith | Free tier (tracing) | smith.langchain.com |

---

## Current Status

- [x] Plan created
- [ ] Phase 1: CLI agent (Week 1)
- [ ] Phase 2: Error recovery (Week 2)
- [ ] Phase 3: Streamlit UI (Week 3)
- [ ] Phase 4: Visualizations (Week 4)
- [ ] Phase 5: SQL support (Week 5)
- [ ] Phase 6: Evals + Observability (Week 6)
- [ ] Phase 7: BYOK + Multi-model (Week 7)
- [ ] Phase 8: Deploy + Polish (Week 8)
