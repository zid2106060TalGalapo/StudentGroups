from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


PREFERENCE_COLUMNS = [
    "PreferredProject1",
    "PreferredProject2",
    "PreferredProject3",
]

REQUIRED_COLUMNS = [
    "StudentID",
    "Name",
    "Gender",
    "Nationality",
    "Major",
    "Email",
    *PREFERENCE_COLUMNS,
]

PROJECT_REQUIRED_COLUMNS = [
    "project_id",
    "project_name",
    "description",
    "difficulty",
    "min_team_size",
    "max_team_size",
]


@dataclass(frozen=True)
class Student:
    student_id: str
    name: str
    gender: str
    nationality: str
    major: str
    email: str
    preferences: List[str]


@dataclass(frozen=True)
class Project:
    project_id: str
    project_name: str
    description: str
    difficulty: str
    min_team_size: int
    max_team_size: int


@dataclass
class Group:
    project: Project
    capacity: int
    students: List[Student] = field(default_factory=list)

    def gender_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for student in self.students:
            key = normalize_gender(student.gender)
            counts[key] = counts.get(key, 0) + 1
        return counts



def normalize_gender(value: str) -> str:
    text = (value or "").strip()
    return text if text else "Unspecified"
