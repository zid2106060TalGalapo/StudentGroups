from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from student_groups.models import PROJECT_REQUIRED_COLUMNS, Project


@dataclass
class ProjectAssessment:
    project_name: str
    difficulty: str
    min_team_size: int
    max_team_size: int
    size_ok: bool
    reasoning: str


class ProjectContextMCPTool:
    """A tiny MCP-style adapter that exposes project metadata as external tool context."""

    def __init__(self, json_path: Path):
        self.json_path = json_path
        self._projects = self._load_projects(json_path)

    def list_projects(self) -> List[Project]:
        return list(self._projects.values())

    def assess_group(self, project_name: str, group_size: int) -> ProjectAssessment:
        project = self._projects[project_name]
        size_ok = project.min_team_size <= group_size <= project.max_team_size
        if size_ok:
            reasoning = (
                f"Used MCP project context: difficulty is {project.difficulty} and team size {group_size} "
                f"fits the recommended range {project.min_team_size}-{project.max_team_size}."
            )
        elif group_size < project.min_team_size:
            reasoning = (
                f"Used MCP project context: difficulty is {project.difficulty} and team size {group_size} "
                f"is below the recommended minimum of {project.min_team_size}."
            )
        else:
            reasoning = (
                f"Used MCP project context: difficulty is {project.difficulty} and team size {group_size} "
                f"is above the recommended maximum of {project.max_team_size}."
            )
        return ProjectAssessment(
            project_name=project_name,
            difficulty=project.difficulty,
            min_team_size=project.min_team_size,
            max_team_size=project.max_team_size,
            size_ok=size_ok,
            reasoning=reasoning,
        )

    def _load_projects(self, json_path: Path) -> Dict[str, Project]:
        data = json.loads(json_path.read_text(encoding="utf-8-sig"))
        items = data.get("projects", [])
        projects = {}
        for item in items:
            missing = [field for field in PROJECT_REQUIRED_COLUMNS if field not in item]
            if missing:
                raise ValueError(f"Missing project fields in JSON: {', '.join(missing)}")
            project = Project(
                project_id=str(item["project_id"]).strip(),
                project_name=str(item["project_name"]).strip(),
                description=str(item["description"]).strip(),
                difficulty=str(item["difficulty"]).strip(),
                min_team_size=int(item["min_team_size"]),
                max_team_size=int(item["max_team_size"]),
            )
            projects[project.project_name] = project
        if not projects:
            raise ValueError("The projects JSON did not contain any projects.")
        return projects
