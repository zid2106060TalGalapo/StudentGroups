# Student Grouping Optimiser - Documentation

## Overview
This project is an **agentic AI workflow** for generating student project groups based on preferences, fairness, and group-sizing constraints.

The system is built to work even without an LLM (fallback heuristics), but can leverage a local open model via [Ollama](https://ollama.com/) for more human-like goal-setting and reporting.

---

## Key Components

### 1. Data Ingestion
- Reads student records and project offerings from CSV files.
- Expected input formats are described in `README.md`.

### 2. Deterministic Baseline Allocation
- A baseline allocator creates a deterministic assignment of students to groups.
- This ensures reproducibility and provides a reference point.

### 3. Agentic Optimisation
- An **agentic phase** evaluates the baseline and attempts to improve it using non-deterministic exploration.
- The agentic process is guided by goals set by an LLM (if available) or by default heuristics.

### 4. Evaluation and Reporting
- Candidate allocations are scored based on goals (preferences, fairness, size balance).
- The best allocation is selected and a teacher report is generated.
- Draft group emails are created for each group.

---

## Why Agentic AI? Benefits & Limitations

### ✅ Benefits
- **Goal-oriented planning:** The agentic phase can incorporate high-level objectives (e.g., “maximize student preference satisfaction” or “ensure balanced group diversity”).
- **Flexible objective weighting:** Goals can be adjusted dynamically (via prompts) without changing code.
- **Exploratory search:** The system can explore multiple non-deterministic allocations and choose the best, potentially outperforming fixed heuristics.
- **Explainability:** The report explains whether the agent improved on the deterministic baseline and why.

### ⚠️ Limitations
- **Non-deterministic results:** Agentic optimization can yield different allocations per run (unless seeded), which may be undesirable in some settings.
- **LLM dependency (optional):** While the system works without an LLM, the most polished goal-setting and reports rely on a functioning local model (Ollama).
- **Compute cost:** Searching multiple candidate allocations increases runtime, especially for large cohorts.
- **No guarantees:** The system improves over baseline according to its scoring function, but it cannot guarantee optimality or fairness in an absolute sense.

---

## Agentic Flow Diagram

```mermaid
flowchart TD
  A[Start] --> B[Load students & projects]
  B --> C[Deterministic baseline allocation]
  C --> D{Agentic phase enabled?}
  D -->|Yes| E[Set goals (LLM or heuristics)]
  D -->|No| G[Skip to evaluation]
  E --> F[Search alternative allocations]
  F --> G[Evaluate candidates vs baseline]
  G --> H[Pick best allocation]
  H --> I[Generate reports + emails]
  I --> Z[Done]
```

> **Note:** The diagram above is rendered in tools that support Mermaid. If your viewer does not render Mermaid, treat it as a high-level flow outline.

---

## Where to Find the Code
- `app.py`: CLI entrypoint and workflow orchestration
- `student_groups/allocator.py`: contains allocation logic and group sizing constraints
- `student_groups/workflow.py`: workflows and agentic orchestration
- `student_groups/llm.py`: LLM prompt logic (Ollama integration)
- `student_groups/cli.py`: command-line parsing and options

---

## Running the Project
1. (Optional) Run an LLM server: `ollama serve`
2. Run the app:

```bash
python app.py --input sample_data/students.csv --projects sample_data/projects.csv --output-dir output --min-group-size 3 --max-group-size 5
```

---

## Notes on Group Sizing
- `--min-group-size` and `--max-group-size` control the allowable range per group.
- The allocator tries to keep group sizes within the range, but depending on student count and preferences, edge cases may require slight deviation.
