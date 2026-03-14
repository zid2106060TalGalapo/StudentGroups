from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from student_groups.allocator import AllocationGoals, AllocationResult, GroupAllocator
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
class AgenticRun:
    result: AllocationResult
    goals: AllocationGoals
    score: float
    verification_summary: str
    diversity_summary: str
    group_assessments: Dict[str, ProjectAssessment]
    achievement_summary: str
    llm_role_summary: str


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
            "Convert a teacher instruction into optimisation weights for a deterministic student grouping optimiser.\n"
            "Return exactly these lines:\n"
            "preference_weight=<number>\n"
            "fairness_weight=<number>\n"
            "size_weight=<number>\n"
            "search_iterations=<integer between 3 and 8>\n"
            "reasoning=<one short sentence>\n\n"
            f"Teacher instruction: {self.teacher_prompt}\n"
            f"Students={len(students)}, Projects={len(projects)}"
        )
        response = self.llm_client.generate(prompt, system_prompt="You convert teacher intent into concise optimisation weights.")
        if response.used_model:
            parsed = self._parse(response.text)
            if parsed:
                return parsed
        return self._fallback_goals()

    def tune(self, current_goals: AllocationGoals, feedback: str) -> AllocationGoals:
        prompt = (
            "Adjust optimisation weights for the next deterministic allocation attempt.\n"
            "Return exactly these lines:\n"
            "preference_weight=<number>\n"
            "fairness_weight=<number>\n"
            "size_weight=<number>\n"
            "reasoning=<one short sentence>\n\n"
            f"Teacher instruction: {current_goals.teacher_prompt}\n"
            f"Current weights: preference={current_goals.preference_weight:.2f}, fairness={current_goals.fairness_weight:.2f}, size={current_goals.size_weight:.2f}\n"
            f"Verification feedback: {feedback}"
        )
        response = self.llm_client.generate(prompt, system_prompt="You tune weights for the next optimisation attempt.")
        if response.used_model:
            tuned = self._parse_tuning(response.text, current_goals)
            if tuned:
                return tuned
        return self._fallback_tune(current_goals, feedback)

    def _fallback_goals(self) -> AllocationGoals:
        preference_score, fairness_score, size_score = self._prompt_scores(self.teacher_prompt)
        total = preference_score + fairness_score + size_score
        return AllocationGoals(
            teacher_prompt=self.teacher_prompt,
            preference_weight=preference_score / total,
            fairness_weight=fairness_score / total,
            size_weight=size_score / total,
            search_iterations=6,
            reasoning="Derived initial weights from the teacher prompt.",
            used_model=False,
        )

    def _fallback_tune(self, current_goals: AllocationGoals, feedback: str) -> AllocationGoals:
        preference = current_goals.preference_weight
        fairness = current_goals.fairness_weight
        size = current_goals.size_weight
        try:
            metrics = json.loads(feedback)
        except json.JSONDecodeError:
            metrics = {}

        outside_preferences = int(metrics.get("outside_preferences", 0))
        average_preference_rank = float(metrics.get("average_preference_rank", 1.0))
        fairness_score = float(metrics.get("fairness_score", 100.0))
        project_size_mismatches = metrics.get("project_size_mismatches", []) or []

        if outside_preferences > 0 or average_preference_rank > 1.6:
            preference += 0.22
        if fairness_score < 90.0:
            fairness += 0.22
        if project_size_mismatches:
            size += 0.22

        total = preference + fairness + size
        return AllocationGoals(
            teacher_prompt=current_goals.teacher_prompt,
            preference_weight=preference / total,
            fairness_weight=fairness / total,
            size_weight=size / total,
            search_iterations=current_goals.search_iterations,
            reasoning=(
                f"Adjusted weights after reviewing verification feedback: fairness={fairness_score:.2f}, "
                f"avg_rank={average_preference_rank:.2f}, outside_preferences={outside_preferences}, "
                f"project_size_mismatches={len(project_size_mismatches)}."
            ),
            used_model=current_goals.used_model,
        )

    def _prompt_scores(self, text: str) -> tuple[float, float, float]:
        lowered = text.lower()
        preference = 1.0
        fairness = 1.0
        size = 1.0

        preference += self._keyword_weight(lowered, ["preference", "preferences", "first choice", "student choice", "student choices"], 2.5)
        fairness += self._keyword_weight(lowered, ["fairness", "equity", "equitable", "balanced", "diversity", "gender"], 2.5)
        size += self._keyword_weight(lowered, ["size", "team size", "group size", "capacity", "difficulty", "project fit"], 2.0)

        if any(term in lowered for term in ["maximise preference", "maximize preference", "maximise student preferences", "maximize student preferences"]):
            preference += 4.0
        if any(term in lowered for term in ["maximise fairness", "maximize fairness", "maximise equity", "maximize equity"]):
            fairness += 4.0
        if any(term in lowered for term in ["maximise size", "maximize size", "keep groups within size", "respect project size"]):
            size += 4.0

        return preference, fairness, size

    def _keyword_weight(self, text: str, keywords: List[str], weight: float) -> float:
        return sum(weight for keyword in keywords if keyword in text)

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
            search_iterations=max(3, min(8, int(values["search_iterations"]))),
            reasoning=values["reasoning"],
            used_model=True,
        )

    def _parse_tuning(self, text: str, current_goals: AllocationGoals) -> AllocationGoals | None:
        patterns = {
            "preference_weight": r"preference_weight\s*=\s*([0-9]*\.?[0-9]+)",
            "fairness_weight": r"fairness_weight\s*=\s*([0-9]*\.?[0-9]+)",
            "size_weight": r"size_weight\s*=\s*([0-9]*\.?[0-9]+)",
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
            teacher_prompt=current_goals.teacher_prompt,
            preference_weight=preference / total,
            fairness_weight=fairness / total,
            size_weight=size / total,
            search_iterations=current_goals.search_iterations,
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
            1 for group in result.groups.values() if not self.mcp_tool.assess_group(group.project.project_name, len(group.students)).size_ok
        )
        size_score = max(0.0, 100.0 - (general_violations * 20.0) - (project_violations * 40.0))
        return round(
            (goals.preference_weight * preference_score)
            + (goals.fairness_weight * result.fairness_score)
            + (goals.size_weight * size_score),
            2,
        )

    def feedback(self, result: AllocationResult) -> str:
        low_fit = [
            group.project.project_name
            for group in result.groups.values()
            if not self.mcp_tool.assess_group(group.project.project_name, len(group.students)).size_ok
        ]
        return json.dumps({
            "fairness_score": result.fairness_score,
            "average_preference_rank": round(result.average_preference_rank, 2),
            "outside_preferences": result.preference_counts["outside_preferences"],
            "project_size_mismatches": low_fit,
        })

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
            f"{self.min_group_size}-{self.max_group_size}. Using MCP project context, the optimiser achieved {project_fit_count} of {len(assessments)} project-specific fits."
        )
        diversity = "; ".join(
            f"{group.project.project_name}: gender={group.gender_counts()}" for _, group in sorted(result.groups.items())
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
        self.allocator = GroupAllocator(target_group_size=target_group_size, min_group_size=min_group_size, max_group_size=max_group_size)

    def run(self, students: List[Student], projects: List[Project]) -> AgenticRun:
        goals = self.goal_agent.run(students, projects)
        best_result = None
        best_goals = goals
        best_score = float("-inf")
        current_goals = goals

        for _ in range(goals.search_iterations):
            candidate = self.allocator.allocate(students, projects, current_goals, strategy="weighted-deterministic+tuned")
            candidate_score = self.verification_agent.score(candidate, current_goals)
            if candidate_score > best_score:
                best_result = candidate
                best_goals = current_goals
                best_score = candidate_score
            current_goals = self.goal_agent.tune(current_goals, self.verification_agent.feedback(candidate))

        verification_summary, diversity_summary, group_assessments, achievement_summary = self.verification_agent.summarize(best_result)
        return AgenticRun(
            result=best_result,
            goals=best_goals,
            score=best_score,
            verification_summary=verification_summary,
            diversity_summary=diversity_summary,
            group_assessments=group_assessments,
            achievement_summary=achievement_summary,
            llm_role_summary=(
                "The LLM did not construct groups directly. It converted the teacher prompt into initial weights and "
                "then reviewed verification feedback to tune those weights across deterministic iterations."
            ),
        )


class ReportingAgent:
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client

    def create_report(self, run: AgenticRun, output_path: Path) -> None:
        summary = build_allocation_summary(run)
        prompt = (
            "Write a concise teacher-facing report for a deterministic student grouping optimiser with agentic weight tuning.\n"
            "Make the LLM's role explicit: it set initial weights from the teacher prompt and tuned them between iterations after reviewing verification feedback.\n"
            "Include sections: Overview, Teacher Demand, LLM Role, Agent Goals, MCP Reasoning, Performance, Goal Achievement, Diversity, Groups, Notes.\n\n"
            f"{summary}"
        )
        response = self.llm_client.generate(prompt, system_prompt="You are an academic project allocation assistant. Be precise, fair-minded, and concise.")
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
                "Include the assigned project name and its description, and mention that an autonomous optimisation workflow considered preferences, project context, and cohort balance.\n\n"
                f"Project: {group.project.project_name}\n"
                f"Description: {group.project.description}\n"
                f"Students: {', '.join(student.name for student in group.students)}"
            )
            response = self.llm_client.generate(prompt, system_prompt="You draft clear, warm, professional university emails.")
            body = response.text if response.used_model else fallback_email_body(group)
            emails.append({
                "project_id": group.project.project_id,
                "project": project_name,
                "description": group.project.description,
                "to": [student.email for student in group.students],
                "subject": f"Your project group assignment: {project_name}",
                "body": body,
            })
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
            "ProjectID", "ProjectName", "Description", "Difficulty", "ProjectMinTeamSize", "ProjectMaxTeamSize",
            "StudentID", "Name", "Gender", "Nationality", "Major", "Email",
        ])
        for _, group in sorted(result.groups.items()):
            for student in sorted(group.students, key=lambda item: item.name):
                writer.writerow([
                    group.project.project_id, group.project.project_name, group.project.description,
                    group.project.difficulty, group.project.min_team_size, group.project.max_team_size,
                    student.student_id, student.name, student.gender, student.nationality, student.major, student.email,
                ])


def build_allocation_summary(run: AgenticRun) -> str:
    result = run.result
    lines = [
        f"Teacher demand: {run.goals.teacher_prompt}",
        f"LLM role: {run.llm_role_summary}",
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
        "# Teacher Report", "", "## Overview",
        "This allocation was produced by a deterministic optimiser with agentic weight tuning. The deterministic allocator built the groups, while the LLM examined each result, adjusted the weights, and helped select the strongest verified outcome.",
        "", "## Teacher Demand", f"- Teacher prompt: {run.goals.teacher_prompt}",
        "", "## LLM Role", f"- LLM use summary: {run.llm_role_summary}",
        "", "## Agent Goals",
        f"- Goal reasoning: {run.goals.reasoning}",
        f"- Goal weights: preference={run.goals.preference_weight:.2f}, fairness={run.goals.fairness_weight:.2f}, size={run.goals.size_weight:.2f}",
        f"- Search iterations: {run.goals.search_iterations}",
        f"- Goal source: {'LLM-guided' if run.goals.used_model else 'heuristic fallback'}",
        "", "## MCP Reasoning", f"- MCP verification summary: {run.verification_summary}", "", "## Performance",
        f"- Final strategy: {result.strategy}", f"- Selection score: {run.score}", f"- Fairness score: {result.fairness_score}",
        f"- Average preference rank: {result.average_preference_rank:.2f}", f"- First choices satisfied: {result.preference_counts['first_choice']}",
        f"- Second choices satisfied: {result.preference_counts['second_choice']}", f"- Third choices satisfied: {result.preference_counts['third_choice']}",
        f"- No preferred choice available: {result.preference_counts['outside_preferences']}", "", "## Goal Achievement",
        f"- Achievement summary: {run.achievement_summary}", "", "## Diversity", f"- Group diversity summary: {run.diversity_summary}", "", "## Groups",
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
        "This demo uses a deterministic allocator for efficiency and a lightweight agentic loop where the LLM sets and tunes weights across a few iterations to better meet teacher goals and fairness targets.",
    ])
    return "\n".join(lines)


def fallback_email_body(group) -> str:
    names = ", ".join(student.name for student in group.students)
    return (
        f"Hello {names},\n\n"
        f"You have been assigned to the project group for '{group.project.project_name}'.\n\n"
        f"Project description: {group.project.description}\n\n"
        "This allocation was produced by a deterministic optimiser with agentic goal tuning that considered student preferences, project context, equal group sizes, and cohort diversity.\n\n"
        "Please introduce yourselves, arrange an initial meeting, and begin discussing how you would like to approach the project.\n\n"
        "Best regards,\nCourse Team"
    )
