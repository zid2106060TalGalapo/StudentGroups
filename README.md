# Student Grouping Optimiser

A simple agentic AI demo that reads `data/students.csv`, queries project metadata from `data/projects.json` through a small MCP-style tool, allocates students to project groups, and then produces:

- a teacher-facing allocation report
- a CSV of final group assignments
- draft group emails with all student emails included in the `to` field

The demo uses a deterministic allocator for the actual grouping work, while the LLM acts as a lightweight optimisation coach: it sets initial weights from the teacher prompt, reviews verification feedback, adjusts the weights across a few iterations, and helps select the best result.

## Expected input files

`data/students.csv` must contain:

`StudentID,Name,Gender,Nationality,Major,Email,PreferredProject1,PreferredProject2,PreferredProject3`

`data/projects.json` must contain project metadata including:

- `project_id`
- `project_name`
- `description`
- `difficulty`
- `min_team_size`
- `max_team_size`

## Quick start

1. Start Ollama and pull an open model if you want LLM-guided tuning:

```bash
ollama serve
ollama pull llama3.1:8b
```

2. Launch the demo window:

```bash
python app.py
```

3. Or run the workflow directly from the command line:

```bash
python app.py --input data/students.csv --projects data/projects.json --output-dir output --teacher-prompt "Try to maximise student preferences but still maintain fairness."
```

## Demo UI

- Running `python app.py` with no arguments opens a simple desktop demo window.
- The popup lets the teacher choose input files, group-size limits, and the teacher prompt.
- After the run, the right-hand panel shows a professional summary of the allocation, key metrics, group sizes, and the full teacher report.

## Agentic flow

1. `IngestionAgent` loads students.
2. `ProjectContextMCPTool` loads `projects.json` and exposes project metadata as external context.
3. `GoalSettingAgent` uses the LLM to convert the teacher prompt into initial optimisation weights.
4. `GroupAllocator` performs deterministic allocation using those weights.
5. `VerificationAgent` checks the result against project metadata, preference satisfaction, and fairness.
6. `GoalSettingAgent.tune()` uses the LLM again to adjust the weights for the next deterministic attempt.
7. `ReportingAgent` explains both the deterministic allocation result and the LLM's tuning role.

## Notes

- The actual group construction is deterministic for speed and reliability.
- The LLM is used only for goal setting and weight tuning across a few iterations.
- The teacher report includes an explicit `LLM Role` section so the agentic contribution is easy to see.
