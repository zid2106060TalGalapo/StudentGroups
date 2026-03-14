# Student Grouping Optimiser

A simple agentic AI demo that reads a CSV of students, allocates them to project groups with an emphasis on project preference satisfaction and gender balance, and then produces:

- a teacher-facing allocation report
- a CSV of final group assignments
- draft group emails with all student emails included in the `to` field

The workflow is designed for class demos and uses an open LLM through a local [Ollama](https://ollama.com/) server for narrative generation. If Ollama is not running, the app still completes the allocation and falls back to deterministic report and email text.

## Expected CSV format

The input students.csv file must contain these columns:

`StudentID,Name,Gender,Nationality,Major,Email,PreferredProject1,PreferredProject2,PreferredProject3`

The input projects.csv file must contain these columns:

`ProjectID,ProjectName,Description`


## Quick start

1. Create and activate a virtual environment if you want an isolated run.
2. Start Ollama and pull an open model if you want LLM-polished outputs:

```bash
ollama serve
ollama pull llama3.1:8b
```

3. Run the demo:

```bash
python app.py --input sample_data/students.csv --projects sample_data/projects.csv --output-dir output

```

No third-party Python packages are required for the fallback workflow.

## Outputs

The run writes these files into the output directory:

- `allocations.csv`
- `teacher_report.md`
- `group_emails.json`

## Configuration

Optional flags:

- `--group-size 4` to control the approximate target group size and how many project groups are activated
- `--model llama3.1:8b` to choose an Ollama model
- `--ollama-url http://localhost:11434/api/generate` to override the Ollama endpoint

## Agentic flow

The demo uses a lightweight multi-step workflow:

1. `IngestionAgent` validates and parses the student CSV.
2. `AllocationAgent` computes group assignments using preferences and a gender-fairness objective.
3. `ReportingAgent` summarizes the allocation for the teacher.
4. `EmailAgent` drafts one email per group.

The optimiser itself is deterministic. The LLM is used to polish the human-facing communication and explanation layer, which keeps the demo dependable during class.

## Assumptions

- Each distinct preferred project is treated as a candidate project group.
- The system activates the most in-demand projects based on cohort size and the target group size.
- Students are allocated to one of their three stated preferences whenever possible.
- Gender balance is approximated by keeping each group's gender mix close to the cohort-wide gender distribution.

## Demo tip

If you want a fully offline demo without Ollama, simply run the app without a model server. The optimisation still works and the report/email generation uses built-in fallback text.
