from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, List

from student_groups.models import Group, Project, Student, normalize_gender


PREFERENCE_PENALTIES = {0: 0.0, 1: 1.0, 2: 2.0}
UNPREFERRED_PENALTY = 4.0
PROJECT_DEMAND_WEIGHTS = {0: 3, 1: 2, 2: 1}


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
class AllocationResult:
    groups: Dict[str, Group]
    preference_counts: Dict[str, int]
    average_preference_rank: float
    fairness_score: float
    strategy: str = "weighted-deterministic"


class GroupAllocator:
    def __init__(self, target_group_size: int = 4, min_group_size: int = 3, max_group_size: int = 5):
        self.target_group_size = max(2, target_group_size)
        self.min_group_size = max(2, min_group_size)
        self.max_group_size = max(self.min_group_size, max_group_size)

    def allocate(
        self,
        students: List[Student],
        offered_projects: List[Project],
        goals: AllocationGoals,
        strategy: str = "weighted-deterministic",
    ) -> AllocationResult:
        groups = self._build_groups(students, offered_projects, goals)
        cohort_gender_ratio = self._cohort_gender_ratio(students)
        ordered_students = sorted(
            students,
            key=lambda student: (
                -len(set(student.preferences)),
                self._preference_priority(student, goals),
                student.name,
            ),
        )

        for student in ordered_students:
            project_name = self._best_group_for_student(student, groups, cohort_gender_ratio, goals)
            if not project_name:
                raise ValueError("Unable to allocate all students while respecting group size constraints.")
            groups[project_name].students.append(student)

        self._improve_via_swaps(groups, cohort_gender_ratio, goals)
        return self._build_result(groups, students, cohort_gender_ratio, strategy)

    def _build_groups(self, students: List[Student], offered_projects: List[Project], goals: AllocationGoals) -> Dict[str, Group]:
        total_students = len(students)
        desired_group_count = max(1, ceil(total_students / self.target_group_size))
        min_group_count = max(1, ceil(total_students / self.max_group_size))

        demand_scores = {project.project_name: 0 for project in offered_projects}
        for student in students:
            for rank, project_name in enumerate(student.preferences):
                if project_name in demand_scores:
                    demand_scores[project_name] += PROJECT_DEMAND_WEIGHTS.get(rank, 0)

        sorted_projects = sorted(offered_projects, key=lambda item: (-demand_scores[item.project_name], item.project_name))
        best_projects: List[Project] = []
        best_capacities: List[int] = []
        best_score = float("-inf")

        for group_count in range(min_group_count, len(sorted_projects) + 1):
            candidate_projects = sorted_projects[:group_count]
            capacities = self._planned_capacities(candidate_projects, total_students)
            if capacities is None:
                continue
            preference_fit = sum(demand_scores[project.project_name] for project in candidate_projects)
            score = (
                goals.preference_weight * preference_fit * 10.0
                + goals.size_weight * group_count * 10.0
                - abs(group_count - desired_group_count)
            )
            if score > best_score:
                best_score = score
                best_projects = candidate_projects
                best_capacities = capacities

        if not best_projects:
            raise ValueError("No feasible allocation exists within the requested and project group-size limits.")

        groups: Dict[str, Group] = {}
        for project, capacity in zip(best_projects, best_capacities):
            groups[project.project_name] = Group(project=project, capacity=capacity)
        return groups

    def _planned_capacities(self, projects: List[Project], total_students: int) -> List[int] | None:
        minimums = []
        maximums = []
        for project in projects:
            lower = max(self.min_group_size, project.min_team_size)
            upper = min(self.max_group_size, project.max_team_size)
            if lower > upper:
                return None
            minimums.append(lower)
            maximums.append(upper)

        min_total = sum(minimums)
        max_total = sum(maximums)
        if total_students < min_total or total_students > max_total:
            return None

        capacities = list(minimums)
        remaining = total_students - min_total
        while remaining > 0:
            progressed = False
            for index, current in enumerate(capacities):
                if current < maximums[index]:
                    capacities[index] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                return None
        return capacities

    def _cohort_gender_ratio(self, students: List[Student]) -> Dict[str, float]:
        counts: Dict[str, int] = {}
        for student in students:
            gender = normalize_gender(student.gender)
            counts[gender] = counts.get(gender, 0) + 1
        total = max(1, len(students))
        return {gender: count / total for gender, count in counts.items()}

    def _preference_priority(self, student: Student, goals: AllocationGoals) -> float:
        if not student.preferences:
            return UNPREFERRED_PENALTY
        base_rank = sum(index + 1 for index, _ in enumerate(student.preferences)) / len(student.preferences)
        return base_rank / max(goals.preference_weight, 0.05)

    def _best_group_for_student(
        self,
        student: Student,
        groups: Dict[str, Group],
        cohort_gender_ratio: Dict[str, float],
        goals: AllocationGoals,
    ) -> str:
        best_project = ""
        best_cost = float("inf")
        for project_name, group in groups.items():
            if len(group.students) >= group.capacity:
                continue
            cost = self._assignment_cost(student, group, cohort_gender_ratio, project_name, goals)
            if cost < best_cost:
                best_cost = cost
                best_project = project_name
        return best_project

    def _assignment_cost(
        self,
        student: Student,
        group: Group,
        cohort_gender_ratio: Dict[str, float],
        project_name: str,
        goals: AllocationGoals,
    ) -> float:
        if len(group.students) >= group.capacity:
            return float("inf")
        try:
            rank = student.preferences.index(project_name)
            preference_cost = PREFERENCE_PENALTIES[rank] / 2.0
        except ValueError:
            preference_cost = UNPREFERRED_PENALTY / 2.0

        fairness_cost = self._fairness_cost(student, group, cohort_gender_ratio)
        seats_left_after_assignment = group.capacity - (len(group.students) + 1)
        size_cost = seats_left_after_assignment / max(1, group.capacity)

        return (
            goals.preference_weight * preference_cost * 220.0
            + goals.fairness_weight * fairness_cost * 260.0
            + goals.size_weight * size_cost * 40.0
        )

    def _fairness_cost(self, student: Student, group: Group, cohort_gender_ratio: Dict[str, float]) -> float:
        projected_counts = group.gender_counts()
        projected_gender = normalize_gender(student.gender)
        projected_counts[projected_gender] = projected_counts.get(projected_gender, 0) + 1
        projected_size = len(group.students) + 1
        return sum(
            abs((projected_counts.get(gender, 0) / projected_size) - ratio)
            for gender, ratio in cohort_gender_ratio.items()
        )

    def _improve_via_swaps(self, groups: Dict[str, Group], cohort_gender_ratio: Dict[str, float], goals: AllocationGoals) -> None:
        improved = True
        while improved:
            improved = False
            project_names = list(groups.keys())
            for left_index, left_name in enumerate(project_names):
                for right_name in project_names[left_index + 1 :]:
                    left_group = groups[left_name]
                    right_group = groups[right_name]
                    current_cost = self._group_cost(left_group, cohort_gender_ratio, goals) + self._group_cost(right_group, cohort_gender_ratio, goals)
                    best_pair = None
                    best_delta = 0.0
                    for left_student in list(left_group.students):
                        for right_student in list(right_group.students):
                            trial_left = Group(left_group.project, left_group.capacity, [s for s in left_group.students if s != left_student] + [right_student])
                            trial_right = Group(right_group.project, right_group.capacity, [s for s in right_group.students if s != right_student] + [left_student])
                            new_cost = self._group_cost(trial_left, cohort_gender_ratio, goals) + self._group_cost(trial_right, cohort_gender_ratio, goals)
                            delta = current_cost - new_cost
                            if delta > best_delta:
                                best_delta = delta
                                best_pair = (left_student, right_student)
                    if best_pair:
                        left_student, right_student = best_pair
                        left_group.students.remove(left_student)
                        right_group.students.remove(right_student)
                        left_group.students.append(right_student)
                        right_group.students.append(left_student)
                        improved = True
                        break
                if improved:
                    break

    def _group_cost(self, group: Group, cohort_gender_ratio: Dict[str, float], goals: AllocationGoals) -> float:
        preference_penalty = 0.0
        for student in group.students:
            try:
                rank = student.preferences.index(group.project.project_name)
                preference_penalty += PREFERENCE_PENALTIES[rank] / 2.0
            except ValueError:
                preference_penalty += UNPREFERRED_PENALTY / 2.0

        fairness_penalty = 2.0 * (1.0 - self._group_fairness_score(group, cohort_gender_ratio))
        return (
            goals.preference_weight * preference_penalty * 220.0
            + goals.fairness_weight * fairness_penalty * 260.0
        )

    def _build_result(self, groups: Dict[str, Group], students: List[Student], cohort_gender_ratio: Dict[str, float], strategy: str) -> AllocationResult:
        preference_counts = {"first_choice": 0, "second_choice": 0, "third_choice": 0, "outside_preferences": 0}
        preference_rank_total = 0
        student_projects = {}
        fairness_total = 0.0
        active_groups = 0

        for group in groups.values():
            for member in group.students:
                student_projects[member.student_id] = group.project.project_name
            if group.students:
                fairness_total += self._group_fairness_score(group, cohort_gender_ratio)
                active_groups += 1

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

        return AllocationResult(
            groups=groups,
            preference_counts=preference_counts,
            average_preference_rank=preference_rank_total / max(1, len(students)),
            fairness_score=round((100.0 * fairness_total / max(1, active_groups)), 2),
            strategy=strategy,
        )

    def _group_fairness_score(self, group: Group, cohort_gender_ratio: Dict[str, float]) -> float:
        counts = group.gender_counts()
        size = len(group.students)
        if size == 0:
            return 1.0
        distance = sum(abs((counts.get(gender, 0) / size) - ratio) for gender, ratio in cohort_gender_ratio.items())
        return max(0.0, 1.0 - (distance / 2.0))
