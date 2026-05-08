# Agent traces

## What they are

An **agent trace** is a machine-readable record of how an agent behaved over one or more turns: prompts, tool calls, model outputs, errors, and metadata. CLEAR expects traces in a form its **agentic** pipeline can normalize (often **JSON** files on disk).

Eval Hub jobs typically receive traces as files under a directory inside the pod (for example after staging from **object storage**) or via **`parameters.data_dir`** when you run locally.

## MLflow-style traces (shape and expectations)

For **sample MLflow-oriented traces** and the rules CLEAR expects, use IBM’s guide:

**[MLFLOW_TRACING_REQUIREMENTS.md](https://github.com/IBM/CLEAR/blob/main/src/clear_eval/agentic/docs/MLFLOW_TRACING_REQUIREMENTS.md)** in **IBM/CLEAR** (`src/clear_eval/agentic/docs/`).

Summary of what that document covers:

**Requirements**

1. **LLM calls as spans** — Each LLM API call must appear as a span with **`span_type`** in **`{CHAT_MODEL, MODEL, GENERATION}`**.

2. **Named parent spans per graph node** — Each LLM call span should have a parent span named after the graph node that issued it (for example `planner`, `analyst`). *Optional for step analysis:* without it, per-component grouping is weaker, but individual scores still apply.

3. **One trace per agent input** — Each trace should contain one complete agent invocation. *Optional for step analysis:* without it, trajectory views are less meaningful, but individual scores still apply.

4. **Tool schemas** — Use **`bind_tools()`** so tool definitions appear in the trace. Describing tools only in the system prompt is not enough for autologging.

**Autologging**

- **`mlflow.langchain.autolog()`** — Satisfies all three core requirements automatically with patterns like `result = graph.invoke(...)`. Async paths (`ainvoke`, `astream`) can be fragile across LangChain/LangGraph versions; sync **`invoke`** is the reliable path.

- **`mlflow.openai.autolog()`** — Only requirement (1) is automatic. For (2) and (3), wrap invocations and name spans explicitly (for example **`mlflow.start_span`**, **`@mlflow.trace`** on node functions). Node span **`span_type`** must **not** be **`CHAT_MODEL`**, **`MODEL`**, or **`GENERATION`**.

**ReAct vs custom graphs:** Prebuilt ReAct agents often expose a single LLM node; custom **StateGraph** agents with multiple LLM nodes enable richer per-component analysis.

**Optional metadata:** **`intent`** (user question) and **`traj_score`** (0–1 label) can be set via trace tags/metadata; CLEAR searches common field names as described in the IBM doc.

## Samples in this repo

Traces for local demos live under **[`input-traces/`](../input-traces/)** as top-level **`*.json`** files. Point **`parameters.data_dir`** at that folder in your job JSON ([03-local-run.md](03-local-run.md)).

**What is checked in today:** **`tr-0e1ef041647642c958a8aaa1892fdb88.json`** is a single MLflow-exported trace (multi-node style agent, research question about city population density). Add more **`*.json`** files in the same folder if you want longer runs or richer dashboards. Large benchmark dumps are usually kept outside this repo.

## Next

Continue to **[03-local-run.md](03-local-run.md)** — set **`parameters.data_dir`** to your trace directory, set **`EVALHUB_JOB_SPEC_PATH`** to a job JSON, and run **`python main.py`** from **`adapters/clear`**.
