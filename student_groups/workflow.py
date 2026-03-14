from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from student_groups.allocator import AllocationResult, GroupAllocator
from student_groups.llm import OllamaClient
from student_groups.models import PREFERENCE_COLUMNS, PROJECT_REQUIRED_COLUMNS, REQUIRED_COLUMNS, Project, Student


@dataclass
class WorkflowOutputs:
    allocations_path: Path
    report_path: Path
    emails_path: Path


class IngestionAgent:
    def load_students(self, csv_path: Path) -> List[Student]:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            self._validate_columns(reader.fieldnames or [], REQUIRED_COLUMNS)
            students = []
            for row in reader:
                preferences = [row[column].strip() for column in PREFERENCE_COLUMNS if row[column].strip()]
                students.append(
                    Student(
                        student_id=row["StudentID"].strip(),
                        name=row["Name"].strip(),
                        gender=row["Gender"].strip(),
                        nationality=row["Nationality"].strip(),
                        major=row["Major"].strip(),
                        email=row["Email"].strip(),
                        preferences=preferences,
                    )
                )
        if not students:
            raise ValueError("The input CSV did not contain any students.")
        return students

    def load_projects(self, csv_path: Path) -> List[Project]:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            self._validate_columns(reader.fieldnames or [], PROJECT_REQUIRED_COLUMNS)
            projects = []
            for row in reader:
                description_parts = [row["Description"].strip()]
                if None in row and row[None]:
                    description_parts.extend(part.strip() for part in row[None] if part.strip())
                description = ", ".join(part for part in description_parts if part)
                if not row["ProjectName"].strip():
                    continue
                projects.append(
                    Project(
                        project_id=row["ProjectID"].strip(),
                        project_name=row["ProjectName"].strip(),
                        description=description,
                    )
                )
        if not projects:
            raise ValueError("The projects CSV did not contain any projects.")
        return projects

    def _validate_columns(self, columns: List[str], required_columns: List[str]) -> None:
        missing = [column for column in required_columns if column not in columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")


class AllocationAgent:
    def __init__(self, target_group_size: int):
        self.allocator = GroupAllocator(target_group_size=target_group_size)

    def run(self, students: List[Student], projects: List[Project]) -> AllocationResult:
        return self.allocator.allocate(students, projects)


class ReportingAgent:
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client

    def create_report(self, result: AllocationResult, output_path: Path) -> None:
        summary = build_allocation_summary(result)
        prompt = (
            "Write a concise teacher-facing report for a student group allocation task.\n"
            "Explain how the allocation balanced project preferences, equal group sizes, and gender fairness.\n"
            "Mention that not all offered projects had to be used.\n"
            "Use markdown with sections: Overview, Allocation Quality, Groups, Notes.\n\n"
            f"{summary}"
        )
        system_prompt = (
            "You are an academic project allocation assistant. "
            "Be precise, fair-minded, and concise."
        )
        response = self.llm_client.generate(prompt, system_prompt=system_prompt)
        report = response.text if response.used_model else fallback_teacher_report(result)
        output_path.write_text(report, encoding="utf-8")


class EmailAgent:
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client

    def create_emails(self, result: AllocationResult, output_path: Path) -> None:
        emails = []
        for project_name, group in sorted(result.groups.items()):
            prompt = (
                "Write a short, friendly email to a student project group.\n"
                "Include the assigned project name and its description, mention that the grouping considered "
                "project preferences and cohort balance, and encourage the team to meet soon.\n\n"
                f"Project: {group.project.project_name}\n"
                f"Description: {group.project.description}\n"
                f"Students: {', '.join(student.name for student in group.students)}\n"
                f"Majors: {', '.join(sorted({student.major for student in group.students}))}"
            )
            system_prompt = "You draft clear, warm, professional university emails."
            response = self.llm_client.generate(prompt, system_prompt=system_prompt)
            body = response.text if response.used_model else fallback_email_body(group)
            emails.append(
                {
                    "project_id": group.project.project_id,
                    "project": project_name,
                    "description": group.project.description,
                    "to": [student.email for student in group.students],
                    "subject": f"Your project group assignment: {project_name}",
                    "body": body,
                }
            )

        output_path.write_text(json.dumps(emails, indent=2), encoding="utf-8")


class StudentGroupingWorkflow:
    def __init__(self, target_group_size: int, model: str, ollama_url: str):
        llm_client = OllamaClient(model=model, url=ollama_url)
        self.ingestion_agent = IngestionAgent()
        self.allocation_agent = AllocationAgent(target_group_size=target_group_size)
        self.reporting_agent = ReportingAgent(llm_client)
        self.email_agent = EmailAgent(llm_client)

    def run(self, input_csv: Path, output_dir: Path, projects_csv: Optional[Path] = None) -> WorkflowOutputs:
        resolved_projects_csv = projects_csv or (input_csv.parent / "projects.csv")
        students = self.ingestion_agent.load_students(input_csv)
        projects = self.ingestion_agent.load_projects(resolved_projects_csv)
        result = self.allocation_agent.run(students, projects)

        output_dir.mkdir(parents=True, exist_ok=True)
        allocations_path = output_dir / "allocations.csv"
        report_path = output_dir / "teacher_report.md"
        emails_path = output_dir / "group_emails.json"

        write_allocations(result, allocations_path)
        self.reporting_agent.create_report(result, report_path)
        self.email_agent.create_emails(result, emails_path)

        return WorkflowOutputs(
            allocations_path=allocations_path,
            report_path=report_path,
            emails_path=emails_path,
        )



def write_allocations(result: AllocationResult, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ProjectID",
                "ProjectName",
                "Description",
                "StudentID",
                "Name",
                "Gender",
                "Nationality",
                "Major",
                "Email",
            ]
        )
        for _, group in sorted(result.groups.items()):
            for student in sorted(group.students, key=lambda item: item.name):
                writer.writerow(
                    [
                        group.project.project_id,
                        group.project.project_name,
                        group.project.description,
                        student.student_id,
                        student.name,
                        student.gender,
                        student.nationality,
                        student.major,
                        student.email,
                    ]
                )



def build_allocation_summary(result: AllocationResult) -> str:
    lines = [
        f"Fairness score: {result.fairness_score}",
        f"Average preference rank: {result.average_preference_rank:.2f}",
        f"Preference counts: {result.preference_counts}",
    ]
    for _, group in sorted(result.groups.items()):
        lines.append(
            f"Group {group.project.project_name} ({group.project.project_id}): {len(group.students)} students, "
            f"genders={group.gender_counts()}, description={group.project.description}, "
            f"members={[student.name for student in group.students]}"
        )
    return "\n".join(lines)



def fallback_teacher_report(result: AllocationResult) -> str:
    lines = [
        "# Teacher Report",
        "",
        "## Overview",
        (
            f"The optimiser created {len(result.groups)} project groups from the offered project list while balancing "
            f"student preferences, equal group sizes, and gender distribution across the cohort."
        ),
        "",
        "## Allocation Quality",
        f"- Fairness score: {result.fairness_score}",
        f"- Average preference rank: {result.average_preference_rank:.2f}",
        f"- First choices satisfied: {result.preference_counts['first_choice']}",
        f"- Second choices satisfied: {result.preference_counts['second_choice']}",
        f"- Third choices satisfied: {result.preference_counts['third_choice']}",
        f"- Randomly allocated outside preferences: {result.preference_counts['outside_preferences']}",
        "",
        "## Groups",
    ]

    for _, group in sorted(result.groups.items()):
        majors = ", ".join(sorted({student.major for student in group.students}))
        nationalities = ", ".join(sorted({student.nationality for student in group.students}))
        lines.extend(
            [
                f"### {group.project.project_name} ({group.project.project_id})",
                f"- Description: {group.project.description}",
                f"- Students: {', '.join(student.name for student in group.students)}",
                f"- Group size: {len(group.students)}",
                f"- Gender mix: {group.gender_counts()}",
                f"- Majors represented: {majors}",
                f"- Nationalities represented: {nationalities}",
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            (
                "Projects were chosen from the offered project catalogue. High-demand preferred projects were used first, "
                "and any remaining students were assigned across active groups to preserve equal sizes and similar gender diversity."
            ),
        ]
    )
    return "\n".join(lines)



def fallback_email_body(group) -> str:
    names = ", ".join(student.name for student in group.students)
    return (
        f"Hello {names},\n\n"
        f"You have been assigned to the project group for '{group.project.project_name}'.\n\n"
        f"Project description: {group.project.description}\n\n"
        "This allocation considered student project preferences while also aiming for equal-sized groups and a fair distribution across the cohort.\n\n"
        "Please introduce yourselves, arrange an initial meeting, and begin discussing how you would like to approach the project.\n\n"
        "Best regards,\nCourse Team"
    )
