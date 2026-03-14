# Student Grouping Optimiser

A simple agentic AI demo that reads a CSV of students, allocates them to project groups, and then produces:

- a teacher-facing allocation report
- a CSV of final group assignments
- draft group emails with all student emails included in the `to` field

The workflow keeps a deterministic baseline allocator for reproducibility, then adds a lightweight agentic optimisation phase that sets goals, explores non-deterministic alternatives, and keeps the best result. An open LLM through local [Ollama](https://ollama.com/) can guide the goal-setting and reporting steps. If Ollama is not running, the app still completes the workflow with heuristic fallback logic.

## Expected CSV format

The input students.csv file must contain these columns:

`StudentID,Name,Gender,Nationality,Major,Email,PreferredProject1,PreferredProject2,PreferredProject3`

The input projects.csv file must contain these columns:

`ProjectID,ProjectName,Description`

## Quick start

1. Start Ollama and pull an open model if you want LLM-guided goals and polished outputs:

```bash
ollama serve
ollama pull llama3.1:8b
```

2. Run the demo:

```bash
python app.py --input sample_data/students.csv --projects sample_data/projects.csv --output-dir output --min-group-size 3 --max-group-size 5
```

You can adjust `--min-group-size` and `--max-group-size` to control the range of group sizes (default is 4-6).

No third-party Python packages are required for the fallback workflow.

## Agentic flow

1. `IngestionAgent` loads students and offered projects.
2. `GoalSettingAgent` decides optimisation priorities for preferences, fairness, and equal group size.
3. `AllocationAgent` builds a deterministic baseline, then runs a non-deterministic search over alternative allocations.
4. `EvaluationAgent` scores candidates and keeps the best result.
5. `ReportingAgent` explains the final allocation and reports any improvement over baseline.
6. `EmailAgent` drafts one email per group.

## Notes

- The deterministic baseline remains the reference point.
- The agentic phase does not blindly replace it. It must beat the baseline according to its stated goals.
- Not all offered projects need to be used.
