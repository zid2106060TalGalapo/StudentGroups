from __future__ import annotations

import argparse
from pathlib import Path

from student_groups.workflow import DEFAULT_TEACHER_PROMPT, StudentGroupingWorkflow



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Student grouping optimiser demo")
    parser.add_argument("--input", default="data/students.csv", help="Path to the student CSV file")
    parser.add_argument("--projects", default="data/projects.json", help="Path to the offered projects JSON file")
    parser.add_argument("--output-dir", default="output", help="Directory for generated outputs")
    parser.add_argument("--group-size", type=int, default=4, help="Approximate target group size")
    parser.add_argument("--min-group-size", type=int, default=3, help="Minimum allowed group size")
    parser.add_argument("--max-group-size", type=int, default=5, help="Maximum allowed group size")
    parser.add_argument(
        "--teacher-prompt",
        default=DEFAULT_TEACHER_PROMPT,
        help="Teacher instruction prompt used to derive dynamic optimisation weights",
    )
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model name")
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/generate",
        help="Ollama generate endpoint",
    )
    return parser



def main() -> int:
    args = build_parser().parse_args()
    workflow = StudentGroupingWorkflow(
        target_group_size=args.group_size,
        min_group_size=args.min_group_size,
        max_group_size=args.max_group_size,
        teacher_prompt=args.teacher_prompt,
        model=args.model,
        ollama_url=args.ollama_url,
    )
    outputs = workflow.run(
        input_csv=Path(args.input),
        projects_csv=Path(args.projects) if args.projects else None,
        output_dir=Path(args.output_dir),
    )

    print("Student grouping workflow completed.")
    print(f"Allocations: {outputs.allocations_path}")
    print(f"Teacher report: {outputs.report_path}")
    print(f"Group emails: {outputs.emails_path}")
    return 0
