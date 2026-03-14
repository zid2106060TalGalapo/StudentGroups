# Student Grouping Optimiser

A simple agentic AI demo that reads `data/students.csv`, queries project metadata from `data/projects.json` through a small MCP-style tool, allocates students to project groups, and then produces:

- a teacher-facing allocation report
- a CSV of final group assignments
- draft group emails with all student emails included in the `to` field

The workflow keeps the deterministic allocator in code as a reference path, but the demo focuses on the agentic layer: the agent sets goals, calls an MCP-style project context tool, explores candidate allocations, verifies project fit, and selects the final result. An open LLM through local [Ollama](https://ollama.com/) can guide the goal-setting and reporting steps. If Ollama is not running, the app still completes the workflow with heuristic fallback logic.

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

1. Start Ollama and pull an open model if you want LLM-guided goals and polished outputs:

```bash
ollama serve
ollama pull llama3.1:8b
```

2. Run the demo:

```bash
python app.py --input data/students.csv --projects data/projects.json --output-dir output --min-group-size 3 --max-group-size 5
```

## Agentic flow

1. `IngestionAgent` loads students.
2. `ProjectContextMCPTool` loads `projects.json` and exposes project metadata as external context.
3. `GoalSettingAgent` decides optimisation priorities.
4. `AllocationAgent` generates candidate allocations while prioritising groups that stay inside each selected project's MCP min-max team-size range.
5. `VerificationAgent` queries the MCP tool to check whether each final group fits the project's difficulty and recommended team size.
6. `ReportingAgent` explains the final allocation and the reasoning for each group.
7. `EmailAgent` drafts one email per group.

## Notes

- The MCP-style tool keeps the demo concise while still showing external tool access during reasoning.
- Not all offered projects need to be used.
- The teacher report includes per-group reasoning based on project difficulty and recommended team size.
