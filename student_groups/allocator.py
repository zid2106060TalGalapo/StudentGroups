from __future__ import annotations

import random
from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Tuple

from student_groups.models import Group, Project, Student, normalize_gender


PREFERENCE_WEIGHTS = {0: 0, 1: 15, 2: 35}
UNPREFERRED_PENALTY = 120
CAPACITY_PENALTY = 2000
GENDER_BALANCE_WEIGHT = 60.0
PROJECT_DEMAND_WEIGHTS = {0: 3, 1: 2, 2: 1}
TIE_BREAK_EPSILON = 1e-9


@dataclass
class AllocationResult:
    groups: Dict[str, Group]
    preference_counts: Dict[str, int]
    average_preference_rank: float
    fairness_score: float


class GroupAllocator:
    def __init__(self, target_group_size: int = 4, seed: int = 42):
        self.target_group_size = max(2, target_group_size)
        self.random = random.Random(seed)

    def allocate(self, students: List[Student], offered_projects: List[Project]) -> AllocationResult:
        groups = self._build_groups(students, offered_projects)
        cohort_gender_ratio = self._cohort_gender_ratio(students)

        ordered_students = sorted(
            students,
            key=lambda student: self._student_priority(student),
            reverse=True,
        )

        for student in ordered_students:
            project_name = self._best_group_for_student(student, groups, cohort_gender_ratio)
            groups[project_name].students.append(student)

        self._improve_via_swaps(groups, cohort_gender_ratio)
        return self._build_result(groups, students, cohort_gender_ratio)

    def _build_groups(self, students: List[Student], offered_projects: List[Project]) -> Dict[str, Group]:
        if not offered_projects:
            raise ValueError("The projects CSV did not contain any projects.")

        total_students = len(students)
        desired_group_count = max(1, ceil(total_students / self.target_group_size))
        active_group_count = min(len(offered_projects), desired_group_count)

        demand_scores = {project.project_name: 0 for project in offered_projects}
        for student in students:
            for rank, project_name in enumerate(student.preferences):
                if project_name in demand_scores:
                    demand_scores[project_name] += PROJECT_DEMAND_WEIGHTS.get(rank, 0)

        sorted_projects = sorted(
            offered_projects,
            key=lambda item: (-demand_scores[item.project_name], item.project_name),
        )
        active_projects = sorted_projects[:active_group_count]

        base_capacity = total_students // active_group_count
        remainder = total_students % active_group_count

        groups: Dict[str, Group] = {}
        for index, project in enumerate(active_projects):
            capacity = base_capacity + (1 if index < remainder else 0)
            groups[project.project_name] = Group(project=project, capacity=max(1, capacity))
        return groups

    def _cohort_gender_ratio(self, students: List[Student]) -> Dict[str, float]:
        counts: Dict[str, int] = {}
        for student in students:
            gender = normalize_gender(student.gender)
            counts[gender] = counts.get(gender, 0) + 1
        total = max(1, len(students))
        return {gender: count / total for gender, count in counts.items()}

    def _student_priority(self, student: Student) -> Tuple[int, int]:
        distinct_preferences = len(set(student.preferences))
        unspecified_gender = 1 if normalize_gender(student.gender) == "Unspecified" else 0
        return (distinct_preferences, -unspecified_gender)

    def _best_group_for_student(
        self,
        student: Student,
        groups: Dict[str, Group],
        cohort_gender_ratio: Dict[str, float],
    ) -> str:
        best_cost = float("inf")
        best_projects: List[str] = []

        for project_name, group in groups.items():
            cost = self._assignment_cost(student, group, cohort_gender_ratio, project_name)
            if cost + TIE_BREAK_EPSILON < best_cost:
                best_cost = cost
                best_projects = [project_name]
            elif abs(cost - best_cost) <= TIE_BREAK_EPSILON:
                best_projects.append(project_name)

        return self.random.choice(best_projects)

    def _assignment_cost(
        self,
        student: Student,
        group: Group,
        cohort_gender_ratio: Dict[str, float],
        project_name: str,
    ) -> float:
        try:
            preference_rank = student.preferences.index(project_name)
            preference_cost = PREFERENCE_WEIGHTS[preference_rank]
        except ValueError:
            preference_cost = UNPREFERRED_PENALTY

        capacity_cost = CAPACITY_PENALTY if len(group.students) >= group.capacity else 0
        fairness_cost = self._gender_fairness_cost(student, group, cohort_gender_ratio)
        return preference_cost + capacity_cost + fairness_cost

    def _gender_fairness_cost(
        self,
        student: Student,
        group: Group,
        cohort_gender_ratio: Dict[str, float],
    ) -> float:
        candidate_gender = normalize_gender(student.gender)
        projected_size = len(group.students) + 1
        projected_counts = group.gender_counts()
        projected_counts[candidate_gender] = projected_counts.get(candidate_gender, 0) + 1

        distance = 0.0
        for gender, target_ratio in cohort_gender_ratio.items():
            actual_ratio = projected_counts.get(gender, 0) / projected_size
            distance += abs(actual_ratio - target_ratio)
        return distance * GENDER_BALANCE_WEIGHT

    def _improve_via_swaps(
        self,
        groups: Dict[str, Group],
        cohort_gender_ratio: Dict[str, float],
    ) -> None:
        improved = True
        while improved:
            improved = False
            projects = list(groups.keys())
            for left_index, left_project in enumerate(projects):
                for right_project in projects[left_index + 1 :]:
                    left_group = groups[left_project]
                    right_group = groups[right_project]
                    swap = self._find_better_swap(left_group, right_group, cohort_gender_ratio)
                    if not swap:
                        continue
                    left_student, right_student = swap
                    left_group.students.remove(left_student)
                    right_group.students.remove(right_student)
                    left_group.students.append(right_student)
                    right_group.students.append(left_student)
                    improved = True
                    break
                if improved:
                    break

    def _find_better_swap(
        self,
        left_group: Group,
        right_group: Group,
        cohort_gender_ratio: Dict[str, float],
    ) -> Tuple[Student, Student] | None:
        current_cost = self._group_cost(left_group, cohort_gender_ratio) + self._group_cost(
            right_group, cohort_gender_ratio
        )
        best_improvement = 0.0
        best_pair: Tuple[Student, Student] | None = None

        for left_student in list(left_group.students):
            for right_student in list(right_group.students):
                new_left = [student for student in left_group.students if student != left_student] + [right_student]
                new_right = [student for student in right_group.students if student != right_student] + [left_student]

                trial_left = Group(project=left_group.project, capacity=left_group.capacity, students=new_left)
                trial_right = Group(project=right_group.project, capacity=right_group.capacity, students=new_right)
                new_cost = self._group_cost(trial_left, cohort_gender_ratio) + self._group_cost(
                    trial_right, cohort_gender_ratio
                )
                improvement = current_cost - new_cost
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_pair = (left_student, right_student)

        return best_pair

    def _group_cost(self, group: Group, cohort_gender_ratio: Dict[str, float]) -> float:
        total = 0.0
        for student in group.students:
            total += self._assignment_cost(student, group, cohort_gender_ratio, group.project.project_name)
        return total

    def _build_result(
        self,
        groups: Dict[str, Group],
        students: List[Student],
        cohort_gender_ratio: Dict[str, float],
    ) -> AllocationResult:
        preference_counts = {"first_choice": 0, "second_choice": 0, "third_choice": 0, "outside_preferences": 0}
        preference_rank_total = 0

        student_projects = {}
        for group in groups.values():
            for member in group.students:
                student_projects[member.student_id] = group.project.project_name

        for student in students:
            project_name = student_projects.get(student.student_id)
            try:
                rank = student.preferences.index(project_name)
                preference_rank_total += rank + 1
                if rank == 0:
                    preference_counts["first_choice"] += 1
                elif rank == 1:
                    preference_counts["second_choice"] += 1
                else:
                    preference_counts["third_choice"] += 1
            except ValueError:
                preference_rank_total += 4
                preference_counts["outside_preferences"] += 1

        fairness_score = 100.0 - min(
            100.0,
            sum(self._group_gender_gap(group, cohort_gender_ratio) for group in groups.values()) * 100.0,
        )
        average_rank = preference_rank_total / max(1, len(students))
        return AllocationResult(
            groups=groups,
            preference_counts=preference_counts,
            average_preference_rank=average_rank,
            fairness_score=round(fairness_score, 2),
        )

    def _group_gender_gap(self, group: Group, cohort_gender_ratio: Dict[str, float]) -> float:
        if not group.students:
            return 0.0
        counts = group.gender_counts()
        size = len(group.students)
        return sum(abs((counts.get(gender, 0) / size) - ratio) for gender, ratio in cohort_gender_ratio.items())
