from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from student_groups.allocator import AllocationResult, GroupAllocator
from student_groups.llm import OllamaClient
from student_groups.models import PREFERENCE_COLUMNS, REQUIRED_COLUMNS, Project, Student
from student_groups.project_context_mcp import ProjectAssessment, ProjectContextMCPTool


DEFAULT_TEACHER_PROMPT = "Try to maximise student preferences but still maintain fairness."


@dataclass
class WorkflowOutputs:
    allocations_path: Path
    report_path: Path
    emails_path: Path


@dataclass
class AllocationGoals:
    teacher_prompt: str
    preference_weight: float
    fairness_weight: float
    size_weight: float
    search_iterations: int
    reasoning: str
    used_model: bool


@dataclass
class AgenticRun:
    result: AllocationResult
    goals: AllocationGoals
    score: float
    verification_summary: str
    diversity_summary: str
    group_assessments: Dict[str, ProjectAssessment]
    achievement_summary: str


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

    def _validate_columns(self, columns: List[str], required_columns: List[str]) -> None:
        missing = [column for column in required_columns if column not in columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")


class GoalSettingAgent:
    def __init__(self, llm_client: OllamaClient, teacher_prompt: str):
        self.llm_client = llm_client
        self.teacher_prompt = teacher_prompt or DEFAULT_TEACHER_PROMPT

    def run(self, students: List[Student], projects: List[Project]) -> AllocationGoals:
        prompt = (
            "Convert a teacher instruction into optimisation weights for an autonomous student grouping agent that can use an MCP tool for project context.\n"
            "Return exactly these lines:\n"
            "preference_weight=<number>\n"
            "fairness_weight=<number>\n"
            "size_weight=<number>\n"
            "search_iterations=<integer between 8 and 30>\n"
            "reasoning=<one short sentence>\n\n"
            f"Teacher instruction: {self.teacher_prompt}\n"
            f"Students={len(students)}, Projects={len(projects)}"
        )
        response = self.llm_client.generate(prompt, system_prompt="You convert teacher intent into concise optimisation goals.")
        if response.used_model:
            parsed = self._parse(response.text)
            if parsed:
                return parsed
        return self._fallback_goals()

    def _fallback_goals(self) -> AllocationGoals:
        text = self.teacher_prompt.lower()
        preference = 0.34
        fairness = 0.33
        size = 0.33
        if any(word in text for word in ["prefer", "preference", "choice"]):
            preference += 0.22
        if any(word in text for word in ["fair", "equitable", "balance", "diversity"]):
            fairness += 0.18
        if any(word in text for word in ["size", "team", "difficulty", "min", "max", "large enough"]):
            size += 0.18
        total = preference + fairness + size
        return AllocationGoals(
            teacher_prompt=self.teacher_prompt,
            preference_weight=preference / total,
            fairness_weight=fairness / total,
            size_weight=size / total,
            search_iterations=14,
            reasoning="Derived weights from the teacher prompt and prioritised the strongest requested outcomes.",
            used_model=False,
        )

    def _parse(self, text: str) -> AllocationGoals | None:
        patterns = {
            "preference_weight": r"preference_weight\s*=\s*([0-9]*\.?[0-9]+)",
            "fairness_weight": r"fairness_weight\s*=\s*([0-9]*\.?[0-9]+)",
            "size_weight": r"size_weight\s*=\s*([0-9]*\.?[0-9]+)",
            "search_iterations": r"search_iterations\s*=\s*(\d+)",
            "reasoning": r"reasoning\s*=\s*(.+)",
        }
        values = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                return None
            values[key] = match.group(1).strip()

        preference = float(values["preference_weight"])
        fairness = float(values["fairness_weight"])
        size = float(values["size_weight"])
        total = preference + fairness + size
        if total <= 0:
            return None
        return AllocationGoals(
            teacher_prompt=self.teacher_prompt,
            preference_weight=preference / total,
            fairness_weight=fairness / total,
            size_weight=size / total,
            search_iterations=max(8, min(30, int(values["search_iterations"]))),
            reasoning=values["reasoning"],
            used_model=True,
        )


class VerificationAgent:
    def __init__(self, target_group_size: int, min_group_size: int, max_group_size: int, mcp_tool: ProjectContextMCPTool):
        self.target_group_size = max(2, target_group_size)
        self.min_group_size = max(2, min_group_size)
        self.max_group_size = max(self.min_group_size, max_group_size)
        self.mcp_tool = mcp_tool

    def score(self, result: AllocationResult, goals: AllocationGoals) -> float:
        total_students = sum(len(group.students) for group in result.groups.values())
        if total_students == 0:
            return 0.0
        preference_score = max(0.0, 100.0 - ((result.average_preference_rank - 1.0) / 3.0) * 100.0)
        group_sizes = [len(group.students) for group in result.groups.values() if group.students]
        general_violations = sum(1 for size in group_sizes if size < self.min_group_size or size > self.max_group_size)
        project_violations = sum(
            1
            for group in result.groups.values()
            if not self.mcp_tool.assess_group(group.project.project_name, len(group.students)).size_ok
        )
        size_score = max(0.0, 100.0 - (general_violations * 20.0) - (project_violations * 40.0))
        return round(
            (goals.preference_weight * preference_score)
            + (goals.fairness_weight * result.fairness_score)
            + (goals.size_weight * size_score),
            2,
        )

    def summarize(self, result: AllocationResult) -> tuple[str, str, Dict[str, ProjectAssessment], str]:
        sizes = [len(group.students) for group in result.groups.values() if group.students]
        size_min = min(sizes) if sizes else 0
        size_max = max(sizes) if sizes else 0
        assessments = {
            project_name: self.mcp_tool.assess_group(project_name, len(group.students))
            for project_name, group in result.groups.items()
        }
        project_fit_count = sum(1 for assessment in assessments.values() if assessment.size_ok)
        verification = (
            f"Verified {len(sizes)} active groups with size range {size_min}-{size_max} against requested range "
            f"{self.min_group_size}-{self.max_group_size}. Using MCP project context, the agent prioritised keeping groups inside each project's recommended team-size range and achieved {project_fit_count} of {len(assessments)} project-specific fits."
        )
        diversity = "; ".join(
            f"{group.project.project_name}: gender={group.gender_counts()}"
            for _, group in sorted(result.groups.items())
        )
        achievement = (
            f"Achieved {result.preference_counts['first_choice']} first-choice, {result.preference_counts['second_choice']} second-choice, "
            f"{result.preference_counts['third_choice']} third-choice, and {result.preference_counts['outside_preferences']} outside-preference allocations, "
            f"with fairness score {result.fairness_score} and {project_fit_count}/{len(assessments)} MCP project-fit matches."
        )
        return verification, diversity, assessments, achievement


class AllocationAgent:
    def __init__(self, target_group_size: int, min_group_size: int, max_group_size: int, teacher_prompt: str, llm_client: OllamaClient, mcp_tool: ProjectContextMCPTool):
        self.goal_agent = GoalSettingAgent(llm_client, teacher_prompt)
        self.verification_agent = VerificationAgent(target_group_size, min_group_size, max_group_size, mcp_tool)
        self.deterministic_allocator = GroupAllocator(
            target_group_size=target_group_size,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
        )

    def run(self, students: List[Student], projects: List[Project]) -> AgenticRun:
        goals = self.goal_agent.run(students, projects)
        best_result = None
        best_score = float("-inf")

        for attempt in range(goals.search_iterations):
            strategy = "agentic-search+mcp" if attempt else "deterministic-reference"
            candidate = self.deterministic_allocator.allocate(
                students,
                projects,
                seed=42 + attempt,
                randomize=attempt > 0,
                strategy=strategy,
            )
            candidate_score = self.verification_agent.score(candidate, goals)
            if candidate_score > best_score:
                best_result = candidate
                best_score = candidate_score

        verification_summary, diversity_summary, group_assessments, achievement_summary = self.verification_agent.summarize(best_result)
        return AgenticRun(
            result=best_result,
            goals=goals,
            score=best_score,
            verification_summary=verification_summary,
            diversity_summary=diversity_summary,
            group_assessments=group_assessments,
            achievement_summary=achievement_summary,
        )


class ReportingAgent:
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client

    def create_report(self, run: AgenticRun, output_path: Path) -> None:
        summary = build_allocation_summary(run)
        prompt = (
            "Write a concise teacher-facing report for an agentic student group allocation workflow.\n"
            "Explain that the agent used an MCP tool to access project metadata, converted the teacher prompt into weights, set goals, generated candidate allocations, verified outcomes, and selected the final result.\n"
            "Include sections: Overview, Teacher Demand, Agent Goals, MCP Reasoning, Performance, Goal Achievement, Diversity, Groups, Notes.\n\n"
            f"{summary}"
        )
        system_prompt = "You are an academic project allocation assistant. Be precise, fair-minded, and concise."
        response = self.llm_client.generate(prompt, system_prompt=system_prompt)
        report = response.text if response.used_model else fallback_teacher_report(run)
        output_path.write_text(report, encoding="utf-8")


class EmailAgent:
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client

    def create_emails(self, result: AllocationResult, output_path: Path) -> None:
        emails = []
        for project_name, group in sorted(result.groups.items()):
            prompt = (
                "Write a short, friendly email to a student project group.\n"
                "Include the assigned project name and its description, and mention that an autonomous allocation workflow considered preferences, project context, and cohort balance.\n\n"
                f"Project: {group.project.project_name}\n"
                f"Description: {group.project.description}\n"
                f"Students: {', '.join(student.name for student in group.students)}"
            )
            response = self.llm_client.generate(prompt, system_prompt="You draft clear, warm, professional university emails.")
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
    def __init__(self, target_group_size: int, min_group_size: int, max_group_size: int, teacher_prompt: str, model: str, ollama_url: str, projects_path: Optional[Path] = None):
        llm_client = OllamaClient(model=model, url=ollama_url)
        self.ingestion_agent = IngestionAgent()
        self.projects_path = projects_path
        self.llm_client = llm_client
        self.target_group_size = target_group_size
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.teacher_prompt = teacher_prompt or DEFAULT_TEACHER_PROMPT

    def run(self, input_csv: Path, output_dir: Path, projects_csv: Optional[Path] = None) -> WorkflowOutputs:
        resolved_projects_path = projects_csv or self.projects_path or (input_csv.parent / "projects.json")
        students = self.ingestion_agent.load_students(input_csv)
        mcp_tool = ProjectContextMCPTool(resolved_projects_path)
        projects = mcp_tool.list_projects()
        allocation_agent = AllocationAgent(
            target_group_size=self.target_group_size,
            min_group_size=self.min_group_size,
            max_group_size=self.max_group_size,
            teacher_prompt=self.teacher_prompt,
            llm_client=self.llm_client,
            mcp_tool=mcp_tool,
        )
        reporting_agent = ReportingAgent(self.llm_client)
        email_agent = EmailAgent(self.llm_client)
        run = allocation_agent.run(students, projects)

        output_dir.mkdir(parents=True, exist_ok=True)
        allocations_path = output_dir / "allocations.csv"
        report_path = output_dir / "teacher_report.md"
        emails_path = output_dir / "group_emails.json"

        write_allocations(run.result, allocations_path)
        reporting_agent.create_report(run, report_path)
        email_agent.create_emails(run.result, emails_path)
        return WorkflowOutputs(allocations_path=allocations_path, report_path=report_path, emails_path=emails_path)



def write_allocations(result: AllocationResult, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "ProjectID",
            "ProjectName",
            "Description",
            "Difficulty",
            "ProjectMinTeamSize",
            "ProjectMaxTeamSize",
            "StudentID",
            "Name",
            "Gender",
            "Nationality",
            "Major",
            "Email",
        ])
        for _, group in sorted(result.groups.items()):
            for student in sorted(group.students, key=lambda item: item.name):
                writer.writerow([
                    group.project.project_id,
                    group.project.project_name,
                    group.project.description,
                    group.project.difficulty,
                    group.project.min_team_size,
                    group.project.max_team_size,
                    student.student_id,
                    student.name,
                    student.gender,
                    student.nationality,
                    student.major,
                    student.email,
                ])



def build_allocation_summary(run: AgenticRun) -> str:
    result = run.result
    lines = [
        f"Teacher demand: {run.goals.teacher_prompt}",
        f"Goal reasoning: {run.goals.reasoning}",
        f"Goal weights: preference={run.goals.preference_weight:.2f}, fairness={run.goals.fairness_weight:.2f}, size={run.goals.size_weight:.2f}",
        f"Search iterations: {run.goals.search_iterations}",
        f"Selection score: {run.score}",
        f"Verification: {run.verification_summary}",
        f"Goal achievement: {run.achievement_summary}",
        f"Preference counts: {result.preference_counts}",
        f"Diversity summary: {run.diversity_summary}",
    ]
    for project_name, assessment in sorted(run.group_assessments.items()):
        lines.append(f"MCP assessment for {project_name}: {assessment.reasoning}")
    return "\n".join(lines)



def fallback_teacher_report(run: AgenticRun) -> str:
    result = run.result
    lines = [
        "# Teacher Report",
        "",
        "## Overview",
        "This allocation was produced by an autonomous agentic workflow. The agent used an MCP tool to access project metadata, translated the teacher's instruction into optimisation weights, generated candidate groupings, verified each candidate against those goals, and selected the final allocation.",
        "",
        "## Teacher Demand",
        f"- Teacher prompt: {run.goals.teacher_prompt}",
        "",
        "## Agent Goals",
        f"- Goal reasoning: {run.goals.reasoning}",
        f"- Goal weights: preference={run.goals.preference_weight:.2f}, fairness={run.goals.fairness_weight:.2f}, size={run.goals.size_weight:.2f}",
        f"- Search iterations: {run.goals.search_iterations}",
        f"- Goal source: {'LLM-guided' if run.goals.used_model else 'heuristic fallback'}",
        "",
        "## MCP Reasoning",
        f"- MCP verification summary: {run.verification_summary}",
        "",
        "## Performance",
        f"- Final strategy: {result.strategy}",
        f"- Selection score: {run.score}",
        f"- Fairness score: {result.fairness_score}",
        f"- Average preference rank: {result.average_preference_rank:.2f}",
        f"- First choices satisfied: {result.preference_counts['first_choice']}",
        f"- Second choices satisfied: {result.preference_counts['second_choice']}",
        f"- Third choices satisfied: {result.preference_counts['third_choice']}",
        f"- No preferred choice available: {result.preference_counts['outside_preferences']}",
        "",
        "## Goal Achievement",
        f"- Achievement summary: {run.achievement_summary}",
        "",
        "## Diversity",
        f"- Group diversity summary: {run.diversity_summary}",
        "",
        "## Groups",
    ]
    for project_name, group in sorted(result.groups.items()):
        assessment = run.group_assessments[project_name]
        majors = ", ".join(sorted({student.major for student in group.students}))
        nationalities = ", ".join(sorted({student.nationality for student in group.students}))
        lines.extend([
            f"### {group.project.project_name} ({group.project.project_id})",
            f"- Description: {group.project.description}",
            f"- Difficulty: {group.project.difficulty}",
            f"- Project team size range: {group.project.min_team_size}-{group.project.max_team_size}",
            f"- Students: {', '.join(student.name for student in group.students)}",
            f"- Group size: {len(group.students)}",
            f"- Gender mix: {group.gender_counts()}",
            f"- Majors represented: {majors}",
            f"- Nationalities represented: {nationalities}",
            f"- Agent reasoning: {assessment.reasoning}",
            "",
        ])
    lines.extend([
        "## Notes",
        "This demo showcases MCP-style tool use inside an agentic workflow: the agent queries external project context during reasoning, converts teacher demand into dynamic weights, and reports how well the final allocation achieved those goals.",
    ])
    return "\n".join(lines)



def fallback_email_body(group) -> str:
    names = ", ".join(student.name for student in group.students)
    return (
        f"Hello {names},\n\n"
        f"You have been assigned to the project group for '{group.project.project_name}'.\n\n"
        f"Project description: {group.project.description}\n\n"
        "This allocation was produced by an autonomous workflow that considered student preferences, project context, equal group sizes, and cohort diversity.\n\n"
        "Please introduce yourselves, arrange an initial meeting, and begin discussing how you would like to approach the project.\n\n"
        "Best regards,\nCourse Team"
    )
