from __future__ import annotations

import csv
import json
import re
import tkinter as tk
import tempfile
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List
from urllib.parse import quote

from student_groups.workflow import DEFAULT_TEACHER_PROMPT, StudentGroupingWorkflow, WorkflowOutputs


def launch_demo_app() -> int:
    app = DemoApp()
    app.mainloop()
    return 0


class DemoApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Agentic AI Student Grouping Demo")
        self.geometry("1160x760")
        self.minsize(980, 680)
        self.configure(bg="#eef2f7")

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background="#eef2f7")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Title.TLabel", background="#eef2f7", foreground="#12344d", font=("Segoe UI Semibold", 22))
        style.configure("Subtitle.TLabel", background="#eef2f7", foreground="#52606d", font=("Segoe UI", 10))
        style.configure("CardTitle.TLabel", background="#ffffff", foreground="#102a43", font=("Segoe UI Semibold", 12))
        style.configure("Body.TLabel", background="#ffffff", foreground="#334e68", font=("Segoe UI", 10))
        style.configure("Metric.TLabel", background="#ffffff", foreground="#102a43", font=("Segoe UI Semibold", 18))
        style.configure("Run.TButton", font=("Segoe UI Semibold", 10))

        self.student_path = tk.StringVar(value="data/students.csv")
        self.project_path = tk.StringVar(value="data/projects.json")
        self.output_dir = tk.StringVar(value="output")
        self.group_size = tk.IntVar(value=4)
        self.min_group_size = tk.IntVar(value=3)
        self.max_group_size = tk.IntVar(value=5)
        self.teacher_prompt = tk.StringVar(value=DEFAULT_TEACHER_PROMPT)
        self.model = tk.StringVar(value="llama3.1:8b")
        self.ollama_url = tk.StringVar(value="http://localhost:11434/api/generate")
        self.status_text = tk.StringVar(value="Ready to run the agentic allocation demo.")
        self.student_count_text = tk.StringVar()
        self.group_details: Dict[str, Dict[str, object]] = {}
        self.current_outputs: WorkflowOutputs | None = None
        self.project_catalog = _load_projects_json(Path(self.project_path.get()))
        self.student_catalog = _load_students_csv(Path(self.student_path.get()))

        self._build_layout()
        self._refresh_student_count()

    def _refresh_student_count(self) -> None:
        self.student_count_text.set(f"There are {len(self.student_catalog)} students in the current demo dataset.")

    def _build_layout(self) -> None:
        container = ttk.Frame(self, style="App.TFrame", padding=20)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=7)
        container.columnconfigure(1, weight=10)
        container.rowconfigure(1, weight=1)

        header = ttk.Frame(container, style="App.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 16))
        ttk.Label(header, text="Agentic AI Student Grouping Demo", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Interactive teacher controls on the left, allocation outcomes and group summaries on the right.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))
        ttk.Label(header, textvariable=self.student_count_text, style="Subtitle.TLabel").pack(anchor="w", pady=(2, 0))

        left = ttk.Frame(container, style="Card.TFrame", padding=18)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 14))
        left.columnconfigure(1, weight=1)

        right = ttk.Frame(container, style="Card.TFrame", padding=18)
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        self._build_form(left)
        self._build_results(right)

    def _build_form(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Demo Controls", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(parent, text="Adjust a few settings, then let the agent optimise the groups.", style="Body.TLabel").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(4, 16)
        )

        fields = [
            ("Students CSV", self.student_path, self._choose_student_file),
            ("Projects JSON", self.project_path, self._choose_project_file),
            ("Output Folder", self.output_dir, self._choose_output_dir),
        ]
        row = 2
        for label, variable, command in fields:
            ttk.Label(parent, text=label, style="Body.TLabel").grid(row=row, column=0, sticky="w", pady=6)
            ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8)
            ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, sticky="ew")
            row += 1

        ttk.Button(parent, text="Manage Students", command=self._open_student_manager).grid(row=row, column=1, sticky="ew", padx=8, pady=(0, 10))
        ttk.Button(parent, text="Manage Projects", command=self._open_project_manager).grid(row=row, column=2, sticky="ew", pady=(0, 10))
        row += 1

        numeric_fields = [
            ("Target group size", self.group_size),
            ("Minimum group size", self.min_group_size),
            ("Maximum group size", self.max_group_size),
        ]
        for label, variable in numeric_fields:
            ttk.Label(parent, text=label, style="Body.TLabel").grid(row=row, column=0, sticky="w", pady=6)
            spin = ttk.Spinbox(parent, from_=2, to=12, textvariable=variable, width=8)
            spin.grid(row=row, column=1, sticky="w", padx=8)
            row += 1

        ttk.Label(parent, text="Teacher prompt", style="Body.TLabel").grid(row=row, column=0, sticky="nw", pady=6)
        prompt = tk.Text(parent, height=6, wrap="word", font=("Segoe UI", 10), relief="solid", borderwidth=1)
        prompt.grid(row=row, column=1, columnspan=2, sticky="nsew", padx=8)
        prompt.insert("1.0", self.teacher_prompt.get())
        self.prompt_widget = prompt
        row += 1

        ttk.Label(parent, text="Ollama model", style="Body.TLabel").grid(row=row, column=0, sticky="w", pady=6)
        ttk.Entry(parent, textvariable=self.model).grid(row=row, column=1, columnspan=2, sticky="ew", padx=8)
        row += 1

        ttk.Label(parent, text="Ollama URL", style="Body.TLabel").grid(row=row, column=0, sticky="w", pady=6)
        ttk.Entry(parent, textvariable=self.ollama_url).grid(row=row, column=1, columnspan=2, sticky="ew", padx=8)
        row += 1

        ttk.Button(parent, text="Run Agentic Allocation", style="Run.TButton", command=self._run_demo).grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(18, 8)
        )
        row += 1
        ttk.Label(parent, textvariable=self.status_text, style="Body.TLabel", wraplength=320, justify="left").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(4, 0)
        )

    def _build_results(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Allocation Summary", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(parent, text="A compact summary for classroom demos.", style="Body.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 12))

        metrics = ttk.Frame(parent, style="Card.TFrame")
        metrics.grid(row=2, column=0, sticky="ew")
        for index in range(5):
            metrics.columnconfigure(index, weight=1)
        self.metric_labels: Dict[str, ttk.Label] = {}
        metric_titles = [
            ("students", "Students"),
            ("fairness", "Fairness"),
            ("first_choice", "First Choice"),
            ("outside", "Outside Top 3"),
            ("fit", "Project Fit"),
        ]
        for index, (key, title) in enumerate(metric_titles):
            card = ttk.Frame(metrics, style="Card.TFrame", padding=(8, 6))
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 8, 0))
            ttk.Label(card, text=title, style="Body.TLabel").pack(anchor="w")
            value = ttk.Label(card, text="-", style="Metric.TLabel")
            value.pack(anchor="w", pady=(6, 0))
            self.metric_labels[key] = value

        notebook = ttk.Notebook(parent)
        notebook.grid(row=3, column=0, sticky="nsew", pady=(14, 0))

        groups_tab = ttk.Frame(notebook)
        report_tab = ttk.Frame(notebook)
        notebook.add(groups_tab, text="Groups")
        notebook.add(report_tab, text="Teacher Report")

        group_header = ttk.Frame(groups_tab)
        group_header.pack(fill="x", pady=(0, 8))
        ttk.Label(group_header, text="Double-click a group to inspect students, move a student, and send email.", style="Body.TLabel").pack(side="left")
        ttk.Button(group_header, text="Open Group Details", command=self._open_selected_group).pack(side="right")

        self.group_tree = ttk.Treeview(
            groups_tab,
            columns=("size", "range", "gender", "project"),
            show="headings",
            height=18,
        )
        for column, width in [("size", 70), ("range", 90), ("gender", 180), ("project", 420)]:
            self.group_tree.heading(column, text=column.title())
            self.group_tree.column(column, width=width, anchor="w")
        self.group_tree.pack(fill="both", expand=True)
        self.group_tree.bind("<Double-1>", self._open_group_from_event)

        report_scroll = ttk.Scrollbar(report_tab)
        report_scroll.pack(side="right", fill="y")
        self.report_text = tk.Text(report_tab, wrap="word", font=("Consolas", 10), yscrollcommand=report_scroll.set)
        self.report_text.pack(fill="both", expand=True)
        report_scroll.config(command=self.report_text.yview)

    def _choose_student_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.student_path.set(path)
            self.student_catalog = _load_students_csv(Path(path))
            self._refresh_student_count()
            self.status_text.set("Loaded student list into working memory.")

    def _choose_project_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if path:
            self.project_path.set(path)
            self.project_catalog = _load_projects_json(Path(path))
            self.status_text.set("Loaded project catalog into working memory.")

    def _open_student_manager(self) -> None:
        try:
            StudentManagerDialog(self)
        except Exception as exc:
            messagebox.showerror("Student manager", str(exc))

    def _open_project_manager(self) -> None:
        try:
            ProjectManagerDialog(self)
        except Exception as exc:
            messagebox.showerror("Project manager", str(exc))

    def _choose_output_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def _run_demo(self) -> None:
        self.teacher_prompt.set(self.prompt_widget.get("1.0", "end").strip() or DEFAULT_TEACHER_PROMPT)
        if self.min_group_size.get() > self.max_group_size.get():
            messagebox.showerror("Invalid size range", "Minimum group size cannot be greater than maximum group size.")
            return

        self.status_text.set("Running the agentic allocation workflow...")
        self.update_idletasks()
        try:
            workflow = StudentGroupingWorkflow(
                target_group_size=self.group_size.get(),
                min_group_size=self.min_group_size.get(),
                max_group_size=self.max_group_size.get(),
                teacher_prompt=self.teacher_prompt.get(),
                model=self.model.get(),
                ollama_url=self.ollama_url.get(),
            )
            runtime_projects = _write_runtime_projects_json(self.project_catalog)
            runtime_students = _write_runtime_students_csv(self.student_catalog)
            outputs = workflow.run(
                input_csv=runtime_students,
                projects_csv=runtime_projects,
                output_dir=Path(self.output_dir.get()),
            )
            self.current_outputs = outputs
            self._load_results(outputs)
            self.status_text.set("Allocation complete. Review the teacher summary and groups on the right.")
        except Exception as exc:
            self.status_text.set("Allocation failed. Please review the settings and try again.")
            messagebox.showerror("Agentic Allocation Demo", str(exc))

    def _load_results(self, outputs: WorkflowOutputs) -> None:
        rows, details = _parse_allocation_rows(outputs.allocations_path, outputs.emails_path)
        self.group_details = details
        report = outputs.report_path.read_text(encoding="utf-8")

        fairness_match = re.search(r"Fairness score: ([0-9.]+)", report)
        first_choice_match = re.search(r"First choices satisfied: (\d+)", report)
        outside_match = re.search(r"No preferred choice available: (\d+)", report)
        fit_match = re.search(r"(\d+) of (\d+) project-specific fits", report)

        self.metric_labels["students"].config(text=str(len(self.student_catalog)))
        self.metric_labels["fairness"].config(text=fairness_match.group(1) if fairness_match else "-")
        self.metric_labels["first_choice"].config(text=first_choice_match.group(1) if first_choice_match else "-")
        self.metric_labels["outside"].config(text=outside_match.group(1) if outside_match else "-")
        self.metric_labels["fit"].config(text=f"{fit_match.group(1)}/{fit_match.group(2)}" if fit_match else "-")

        for item in self.group_tree.get_children():
            self.group_tree.delete(item)
        for row in rows:
            self.group_tree.insert("", "end", iid=row["project"], values=(row["size"], row["range"], row["gender"], row["project"]))

        self.report_text.delete("1.0", "end")
        self.report_text.insert("1.0", report)

    def move_student(self, source_project: str, target_project: str, student_id: str) -> None:
        source = self.group_details[source_project]
        target = self.group_details[target_project]
        student = next((item for item in source["students"] if item["student_id"] == student_id), None)
        if not student:
            raise ValueError("Student not found in the selected group.")
        source["students"].remove(student)
        target["students"].append(student)
        source["students"] = sorted(source["students"], key=lambda item: item["name"])
        target["students"] = sorted(target["students"], key=lambda item: item["name"])
        if not self.current_outputs:
            raise ValueError("Output files are not available yet.")
        _write_allocations_csv(self.current_outputs.allocations_path, self.group_details)
        _write_group_emails_json(self.current_outputs.emails_path, self.group_details)
        _write_teacher_report(
            self.current_outputs.report_path,
            self.group_details,
            Path(self.student_path.get()),
            self.teacher_prompt.get(),
            self.min_group_size.get(),
            self.max_group_size.get(),
        )
        self._load_results(self.current_outputs)
        self.status_text.set(f"Moved student to {target_project}. Allocations, emails, and report were refreshed.")

    def _open_group_from_event(self, _event: tk.Event) -> None:
        self._open_selected_group()

    def _open_selected_group(self) -> None:
        selection = self.group_tree.selection()
        if not selection:
            messagebox.showinfo("Group details", "Select a group first.")
            return
        project = selection[0]
        details = self.group_details.get(project)
        if not details:
            messagebox.showinfo("Group details", "No details available for the selected group.")
            return
        GroupDetailsDialog(self, project, details)


class GroupDetailsDialog(tk.Toplevel):
    def __init__(self, parent: DemoApp, project: str, details: Dict[str, object]) -> None:
        super().__init__(parent)
        self.parent_app = parent
        self.project = project
        self.title(f"Group Details - {project}")
        self.geometry("860x560")
        self.configure(bg="#eef2f7")
        self.details = details

        shell = ttk.Frame(self, padding=18)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(2, weight=1)

        ttk.Label(shell, text=project, font=("Segoe UI Semibold", 18), foreground="#12344d").grid(row=0, column=0, sticky="w")
        summary = f"Project ID: {details['project_id']}    Team size: {details['size']}    Recommended range: {details['range']}"
        ttk.Label(shell, text=summary, font=("Segoe UI", 10), foreground="#52606d").grid(row=1, column=0, sticky="w", pady=(4, 12))

        top = ttk.Frame(shell)
        top.grid(row=2, column=0, sticky="nsew")
        top.columnconfigure(0, weight=1)
        top.rowconfigure(1, weight=1)

        ttk.Label(top, text=str(details["description"]), wraplength=780, font=("Segoe UI", 10), foreground="#334e68").grid(row=0, column=0, sticky="w", pady=(0, 12))

        self.tree = ttk.Treeview(
            top,
            columns=("student_id", "name", "gender", "major", "nationality", "email"),
            show="headings",
            height=12,
        )
        headings = [
            ("student_id", "Student ID", 90),
            ("name", "Name", 150),
            ("gender", "Gender", 80),
            ("major", "Major", 130),
            ("nationality", "Nationality", 120),
            ("email", "Email", 220),
        ]
        for column, label, width in headings:
            self.tree.heading(column, text=label)
            self.tree.column(column, width=width, anchor="w")
        self.tree.grid(row=1, column=0, sticky="nsew")

        scroll = ttk.Scrollbar(top, orient="vertical", command=self.tree.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)

        for student in details["students"]:
            self.tree.insert(
                "",
                "end",
                iid=student["student_id"],
                values=(
                    student["student_id"],
                    student["name"],
                    student["gender"],
                    student["major"],
                    student["nationality"],
                    student["email"],
                ),
            )

        footer = ttk.Frame(shell)
        footer.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        ttk.Button(footer, text="Move Selected Student", command=self._move_selected_student).pack(side="left")
        ttk.Button(footer, text="Email This Group", command=self._email_group).pack(side="right")
        ttk.Button(footer, text="Close", command=self.destroy).pack(side="right", padx=(0, 8))

    def _move_selected_student(self) -> None:
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Move student", "Select a student first.")
            return
        options = sorted(name for name in self.parent_app.group_details if name != self.project)
        if not options:
            messagebox.showinfo("Move student", "No other groups are available.")
            return
        MoveStudentDialog(self, self.project, selection[0], options)

    def _email_group(self) -> None:
        recipients = self.details.get("to", [])
        subject = str(self.details.get("subject", "Project group allocation"))
        body = str(self.details.get("body", ""))
        if not recipients:
            messagebox.showinfo("Email group", "No recipient emails were found for this group.")
            return
        mailto = f"mailto:{','.join(recipients)}?subject={quote(subject)}&body={quote(body)}"
        if len(mailto) > 1800:
            self.clipboard_clear()
            self.clipboard_append(", ".join(recipients))
            messagebox.showinfo(
                "Email group",
                "The email draft is too long for a mailto link. The recipient list has been copied to the clipboard instead.",
            )
            return
        webbrowser.open(mailto)


class MoveStudentDialog(tk.Toplevel):
    def __init__(self, parent: GroupDetailsDialog, source_project: str, student_id: str, options: List[str]) -> None:
        super().__init__(parent)
        self.parent_dialog = parent
        self.source_project = source_project
        self.student_id = student_id
        self.title("Move Student")
        self.geometry("480x190")
        self.resizable(False, False)

        frame = ttk.Frame(self, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Move selected student to:", font=("Segoe UI Semibold", 11)).pack(anchor="w")
        self.target = tk.StringVar(value=options[0])
        ttk.Combobox(frame, textvariable=self.target, values=options, state="readonly", width=36).pack(fill="x", pady=(10, 16))
        ttk.Label(frame, text="This updates the allocations, email drafts, and teacher report.", foreground="#52606d").pack(anchor="w")

        footer = ttk.Frame(frame)
        footer.pack(fill="x", pady=(18, 0))
        ttk.Button(footer, text="Cancel", command=self.destroy, width=12).pack(side="right")
        ttk.Button(footer, text="Move Student", command=self._confirm, width=16).pack(side="right", padx=(0, 8))

    def _confirm(self) -> None:
        try:
            self.parent_dialog.parent_app.move_student(self.source_project, self.target.get(), self.student_id)
        except Exception as exc:
            messagebox.showerror("Move student", str(exc))
            return
        self.parent_dialog.destroy()
        self.destroy()


class StudentManagerDialog(tk.Toplevel):
    def __init__(self, parent: DemoApp) -> None:
        super().__init__(parent)
        self.parent_app = parent
        self.title("Manage Students")
        self.geometry("980x460")
        self.students = [dict(item) for item in parent.student_catalog]

        frame = ttk.Frame(self, padding=18)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Students", font=("Segoe UI Semibold", 14)).grid(row=0, column=0, sticky="w")
        self.tree = ttk.Treeview(frame, columns=("id", "name", "gender", "major", "pref1", "pref2", "pref3"), show="headings", height=14)
        for column, label, width in [("id", "Student ID", 90), ("name", "Name", 160), ("gender", "Gender", 80), ("major", "Major", 140), ("pref1", "Pref 1", 130), ("pref2", "Pref 2", 130), ("pref3", "Pref 3", 130)]:
            self.tree.heading(column, text=label)
            self.tree.column(column, width=width, anchor="w")
        self.tree.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

        button_bar = ttk.Frame(frame)
        button_bar.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(button_bar, text="Add Student", command=self._add_student, width=14).pack(side="left")
        ttk.Button(button_bar, text="Edit Student", command=self._edit_student, width=14).pack(side="left", padx=(8, 0))
        ttk.Button(button_bar, text="Remove Student", command=self._remove_student, width=14).pack(side="left", padx=(8, 0))
        ttk.Button(button_bar, text="Close", command=self.destroy, width=12).pack(side="right")

        self._refresh()

    def _refresh(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for student in self.students:
            self.tree.insert("", "end", iid=student["StudentID"], values=(student["StudentID"], student["Name"], student["Gender"], student["Major"], student["PreferredProject1"], student["PreferredProject2"], student["PreferredProject3"]))

    def _selected_student(self) -> dict | None:
        selection = self.tree.selection()
        if not selection:
            return None
        student_id = selection[0]
        return next((item for item in self.students if item["StudentID"] == student_id), None)

    def _add_student(self) -> None:
        StudentEditorDialog(self, None)

    def _edit_student(self) -> None:
        student = self._selected_student()
        if not student:
            messagebox.showinfo("Manage students", "Select a student first.")
            return
        StudentEditorDialog(self, student)

    def _remove_student(self) -> None:
        student = self._selected_student()
        if not student:
            messagebox.showinfo("Manage students", "Select a student first.")
            return
        self.students = [item for item in self.students if item["StudentID"] != student["StudentID"]]
        self.parent_app.student_catalog = [dict(item) for item in self.students]
        self.parent_app._refresh_student_count()
        self._refresh()
        self.parent_app.status_text.set(f"Removed student {student['Name']} from the in-memory student list.")

    def save_student(self, payload: dict, original_id: str | None) -> None:
        if original_id:
            self.students = [item for item in self.students if item["StudentID"] != original_id]
        self.students.append(payload)
        self.students = sorted(self.students, key=lambda item: item["StudentID"])
        self.parent_app.student_catalog = [dict(item) for item in self.students]
        self.parent_app._refresh_student_count()
        self._refresh()
        self.parent_app.status_text.set(f"Saved student {payload['Name']} to the in-memory student list.")


class StudentEditorDialog(tk.Toplevel):
    def __init__(self, parent: StudentManagerDialog, student: dict | None) -> None:
        super().__init__(parent)
        self.parent_dialog = parent
        self.original_id = student["StudentID"] if student else None
        self.title("Student Editor")
        self.geometry("560x500")
        self.resizable(False, False)

        frame = ttk.Frame(self, padding=18)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        fields = [
            ("StudentID", student["StudentID"] if student else ""),
            ("Name", student["Name"] if student else ""),
            ("Gender", student["Gender"] if student else ""),
            ("Nationality", student["Nationality"] if student else ""),
            ("Major", student["Major"] if student else ""),
            ("Email", student["Email"] if student else ""),
            ("PreferredProject1", student["PreferredProject1"] if student else ""),
            ("PreferredProject2", student["PreferredProject2"] if student else ""),
            ("PreferredProject3", student["PreferredProject3"] if student else ""),
        ]
        self.vars: Dict[str, tk.StringVar] = {}
        row = 0
        for label, value in fields:
            self.vars[label] = tk.StringVar(value=value)
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=6)
            ttk.Entry(frame, textvariable=self.vars[label]).grid(row=row, column=1, sticky="ew", padx=(10, 0))
            row += 1

        footer = ttk.Frame(frame)
        footer.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(18, 0))
        ttk.Button(footer, text="Cancel", command=self.destroy, width=12).pack(side="right")
        ttk.Button(footer, text="Save Student", command=self._save, width=14).pack(side="right", padx=(0, 8))

    def _save(self) -> None:
        payload = {key: var.get().strip() for key, var in self.vars.items()}
        if not payload["StudentID"] or not payload["Name"] or not payload["Email"]:
            messagebox.showerror("Student editor", "Student ID, name, and email are required.")
            return
        self.parent_dialog.save_student(payload, self.original_id)
        self.destroy()


class ProjectManagerDialog(tk.Toplevel):
    def __init__(self, parent: DemoApp) -> None:
        super().__init__(parent)
        self.parent_app = parent
        self.title("Manage Projects")
        self.geometry("760x440")
        self.projects = [dict(item) for item in parent.project_catalog]

        frame = ttk.Frame(self, padding=18)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Projects", font=("Segoe UI Semibold", 14)).grid(row=0, column=0, sticky="w")
        self.tree = ttk.Treeview(frame, columns=("id", "name", "difficulty", "range"), show="headings", height=14)
        for column, label, width in [("id", "Project ID", 90), ("name", "Project Name", 260), ("difficulty", "Difficulty", 90), ("range", "Size Range", 100)]:
            self.tree.heading(column, text=label)
            self.tree.column(column, width=width, anchor="w")
        self.tree.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

        button_bar = ttk.Frame(frame)
        button_bar.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(button_bar, text="Add Project", command=self._add_project, width=14).pack(side="left")
        ttk.Button(button_bar, text="Edit Project", command=self._edit_project, width=14).pack(side="left", padx=(8, 0))
        ttk.Button(button_bar, text="Remove Project", command=self._remove_project, width=14).pack(side="left", padx=(8, 0))
        ttk.Button(button_bar, text="Close", command=self.destroy, width=12).pack(side="right")

        self._refresh()

    def _refresh(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for project in self.projects:
            self.tree.insert("", "end", iid=project["project_id"], values=(project["project_id"], project["project_name"], project["difficulty"], f"{project['min_team_size']}-{project['max_team_size']}"))

    def _selected_project(self) -> dict | None:
        selection = self.tree.selection()
        if not selection:
            return None
        project_id = selection[0]
        return next((item for item in self.projects if item["project_id"] == project_id), None)

    def _add_project(self) -> None:
        ProjectEditorDialog(self, None)

    def _edit_project(self) -> None:
        project = self._selected_project()
        if not project:
            messagebox.showinfo("Manage projects", "Select a project first.")
            return
        ProjectEditorDialog(self, project)

    def _remove_project(self) -> None:
        project = self._selected_project()
        if not project:
            messagebox.showinfo("Manage projects", "Select a project first.")
            return
        self.projects = [item for item in self.projects if item["project_id"] != project["project_id"]]
        self.parent_app.project_catalog = [dict(item) for item in self.projects]
        self._refresh()
        self.parent_app.status_text.set(f"Removed project {project['project_name']} from in-memory project catalog.")

    def save_project(self, payload: dict, original_id: str | None) -> None:
        if original_id:
            self.projects = [item for item in self.projects if item["project_id"] != original_id]
        self.projects.append(payload)
        self.projects = sorted(self.projects, key=lambda item: item["project_id"])
        self.parent_app.project_catalog = [dict(item) for item in self.projects]
        self._refresh()
        self.parent_app.status_text.set(f"Saved project {payload['project_name']} to the in-memory project catalog.")


class ProjectEditorDialog(tk.Toplevel):
    def __init__(self, parent: ProjectManagerDialog, project: dict | None) -> None:
        super().__init__(parent)
        self.parent_dialog = parent
        self.original_id = project["project_id"] if project else None
        self.title("Project Editor")
        self.geometry("520x360")
        self.resizable(False, False)

        frame = ttk.Frame(self, padding=18)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        self.project_id = tk.StringVar(value=project["project_id"] if project else "")
        self.project_name = tk.StringVar(value=project["project_name"] if project else "")
        self.description = tk.StringVar(value=project["description"] if project else "")
        self.difficulty = tk.StringVar(value=project["difficulty"] if project else "medium")
        self.min_team_size = tk.IntVar(value=project["min_team_size"] if project else 4)
        self.max_team_size = tk.IntVar(value=project["max_team_size"] if project else 5)

        row = 0
        for label, variable in [("Project ID", self.project_id), ("Project name", self.project_name), ("Description", self.description), ("Difficulty", self.difficulty)]:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=6)
            ttk.Entry(frame, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=(10, 0))
            row += 1
        for label, variable in [("Min team size", self.min_team_size), ("Max team size", self.max_team_size)]:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=6)
            ttk.Spinbox(frame, from_=2, to=12, textvariable=variable, width=8).grid(row=row, column=1, sticky="w", padx=(10, 0))
            row += 1

        footer = ttk.Frame(frame)
        footer.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(18, 0))
        ttk.Button(footer, text="Cancel", command=self.destroy, width=12).pack(side="right")
        ttk.Button(footer, text="Save Project", command=self._save, width=14).pack(side="right", padx=(0, 8))

    def _save(self) -> None:
        if self.min_team_size.get() > self.max_team_size.get():
            messagebox.showerror("Project editor", "Minimum team size cannot be greater than maximum team size.")
            return
        payload = {
            "project_id": self.project_id.get().strip(),
            "project_name": self.project_name.get().strip(),
            "description": self.description.get().strip(),
            "difficulty": self.difficulty.get().strip().lower() or "medium",
            "min_team_size": int(self.min_team_size.get()),
            "max_team_size": int(self.max_team_size.get()),
        }
        if not payload["project_id"] or not payload["project_name"] or not payload["description"]:
            messagebox.showerror("Project editor", "Project ID, name, and description are required.")
            return
        self.parent_dialog.save_project(payload, self.original_id)
        self.destroy()


def _load_students_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_runtime_students_csv(students: List[dict]) -> Path:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    with handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "StudentID", "Name", "Gender", "Nationality", "Major", "Email",
            "PreferredProject1", "PreferredProject2", "PreferredProject3",
        ])
        writer.writeheader()
        for student in students:
            writer.writerow({
                "StudentID": student.get("StudentID", ""),
                "Name": student.get("Name", ""),
                "Gender": student.get("Gender", ""),
                "Nationality": student.get("Nationality", ""),
                "Major": student.get("Major", ""),
                "Email": student.get("Email", ""),
                "PreferredProject1": student.get("PreferredProject1", ""),
                "PreferredProject2": student.get("PreferredProject2", ""),
                "PreferredProject3": student.get("PreferredProject3", ""),
            })
    return Path(handle.name)


def _load_projects_json(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return list(data.get("projects", []))


def _write_runtime_projects_json(projects: List[dict]) -> Path:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    with handle:
        json.dump({"projects": projects}, handle, indent=2)
    return Path(handle.name)


def _write_allocations_csv(path: Path, groups: Dict[str, Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "ProjectID", "ProjectName", "Description", "Difficulty", "ProjectMinTeamSize", "ProjectMaxTeamSize",
            "StudentID", "Name", "Gender", "Nationality", "Major", "Email",
        ])
        for project in sorted(groups):
            group = groups[project]
            min_size, max_size = str(group["range"]).split("-")
            for student in sorted(group["students"], key=lambda item: item["name"]):
                writer.writerow([
                    group["project_id"], project, group["description"], group.get("difficulty", ""), min_size, max_size,
                    student["student_id"], student["name"], student["gender"], student["nationality"], student["major"], student["email"],
                ])


def _write_group_emails_json(path: Path, groups: Dict[str, Dict[str, object]]) -> None:
    payload = []
    for project in sorted(groups):
        group = groups[project]
        students = sorted(group["students"], key=lambda item: item["name"])
        names = ", ".join(student["name"] for student in students)
        body = (
            f"Hello {names},\n\n"
            f"You have been assigned to the project group for '{project}'.\n\n"
            f"Project description: {group['description']}\n\n"
            "This allocation was produced by a deterministic optimiser with agentic goal tuning that considered student preferences, project context, equal group sizes, and cohort diversity.\n\n"
            "Please introduce yourselves, arrange an initial meeting, and begin discussing how you would like to approach the project.\n\n"
            "Best regards,\nCourse Team"
        )
        to = [student["email"] for student in students]
        group["to"] = to
        group["subject"] = f"Your project group assignment: {project}"
        group["body"] = body
        payload.append({
            "project_id": group["project_id"],
            "project": project,
            "description": group["description"],
            "to": to,
            "subject": group["subject"],
            "body": body,
        })
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_teacher_report(path: Path, groups: Dict[str, Dict[str, object]], students_csv: Path, teacher_prompt: str, min_group_size: int, max_group_size: int) -> None:
    student_preferences, cohort_ratio = _load_student_context(students_csv)
    metrics = _calculate_metrics(groups, student_preferences, cohort_ratio, min_group_size, max_group_size)
    lines = [
        "# Teacher Report",
        "",
        "## Overview",
        "This allocation includes a manual teacher adjustment applied through the demo app after the agentic workflow completed. The files and summary below reflect the current group configuration.",
        "",
        "## Teacher Demand",
        f"- Teacher prompt: {teacher_prompt}",
        "",
        "## LLM Role",
        "- LLM use summary: The LLM set and tuned optimisation weights for the initial allocation. A teacher then refined the result manually in the app.",
        "",
        "## MCP Reasoning",
        f"- MCP verification summary: Verified {metrics['active_groups']} active groups with size range {metrics['size_min']}-{metrics['size_max']} against requested range {min_group_size}-{max_group_size}. Using MCP project context, the current allocation achieved {metrics['fit_count']} of {metrics['active_groups']} project-specific fits.",
        "",
        "## Visual Snapshot",
        f"- Fairness              {_ascii_bar(metrics['fairness_score'], 100)} {metrics['fairness_score']:.1f}/100",
        f"- Project size fit      {_ascii_bar(metrics['fit_count'], max(1, metrics['active_groups']))} {metrics['fit_count']}/{metrics['active_groups']} groups",
        f"- First-choice matches  {_ascii_bar(metrics['preference_counts']['first_choice'], max(1, metrics['student_total']))} {metrics['preference_counts']['first_choice']}/{metrics['student_total']}",
        f"- Outside preferences   {_ascii_bar(metrics['student_total'] - metrics['preference_counts']['outside_preferences'], max(1, metrics['student_total']))} {metrics['preference_counts']['outside_preferences']} students outside top 3",
        "",
        "## Performance",
        "- Final strategy: weighted-deterministic+tuned+manual-adjustment",
        f"- Fairness score: {metrics['fairness_score']:.2f}",
        f"- Average preference rank: {metrics['average_rank']:.2f}",
        f"- First choices satisfied: {metrics['preference_counts']['first_choice']}",
        f"- Second choices satisfied: {metrics['preference_counts']['second_choice']}",
        f"- Third choices satisfied: {metrics['preference_counts']['third_choice']}",
        f"- No preferred choice available: {metrics['preference_counts']['outside_preferences']}",
        "",
        "## Diversity",
        f"- Group diversity summary: {metrics['diversity_summary']}",
        "",
        "## Groups",
    ]
    for project in sorted(groups):
        group = groups[project]
        students = sorted(group["students"], key=lambda item: item["name"])
        female = sum(1 for student in students if str(student["gender"]).lower().startswith("f"))
        male = sum(1 for student in students if str(student["gender"]).lower().startswith("m"))
        min_size, max_size = [int(value) for value in str(group["range"]).split("-")]
        size = len(students)
        if min_size <= size <= max_size:
            reasoning = f"Current team size {size} fits the MCP project range {min_size}-{max_size}."
        elif size < min_size:
            reasoning = f"Current team size {size} is below the MCP project minimum of {min_size}."
        else:
            reasoning = f"Current team size {size} is above the MCP project maximum of {max_size}."
        lines.extend([
            f"### {project} ({group['project_id']})",
            f"- Description: {group['description']}",
            f"- Difficulty: {group.get('difficulty', '')}",
            f"- Project team size range: {group['range']}",
            f"- Students: {', '.join(student['name'] for student in students)}",
            f"- Group size: {size}",
            f"- Gender mix: {{'Female': {female}, 'Male': {male}}}",
            f"- Agent reasoning: {reasoning}",
            "",
        ])
    lines.extend([
        "## Notes",
        "This report was refreshed automatically after a manual move in the demo UI so the allocations, group emails, and teacher summary all stay aligned.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_student_context(path: Path) -> tuple[Dict[str, List[str]], Dict[str, float]]:
    preferences: Dict[str, List[str]] = {}
    gender_counts: Dict[str, int] = {}
    total = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            prefs = [row.get(column, "").strip() for column in ["PreferredProject1", "PreferredProject2", "PreferredProject3"] if row.get(column, "").strip()]
            preferences[row["StudentID"].strip()] = prefs
            gender = row["Gender"].strip() or "Unspecified"
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
            total += 1
    return preferences, {gender: count / max(1, total) for gender, count in gender_counts.items()}


def _calculate_metrics(groups: Dict[str, Dict[str, object]], student_preferences: Dict[str, List[str]], cohort_ratio: Dict[str, float], min_group_size: int, max_group_size: int) -> Dict[str, object]:
    counts = {"first_choice": 0, "second_choice": 0, "third_choice": 0, "outside_preferences": 0}
    total_rank = 0
    fairness_total = 0.0
    active_groups = 0
    fit_count = 0
    sizes: List[int] = []
    diversity_parts: List[str] = []
    student_total = 0
    for project in sorted(groups):
        students = groups[project]["students"]
        size = len(students)
        if size == 0:
            continue
        student_total += size
        sizes.append(size)
        active_groups += 1
        min_size, max_size = [int(value) for value in str(groups[project]["range"]).split("-")]
        if min_group_size <= size <= max_group_size and min_size <= size <= max_size:
            fit_count += 1
        gender_counts: Dict[str, int] = {}
        for student in students:
            prefs = student_preferences.get(student["student_id"], [])
            try:
                rank = prefs.index(project)
                total_rank += rank + 1
                counts[["first_choice", "second_choice", "third_choice"][rank]] += 1
            except (ValueError, IndexError):
                total_rank += 4
                counts["outside_preferences"] += 1
            gender = student["gender"] or "Unspecified"
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
        distance = sum(abs((gender_counts.get(gender, 0) / size) - ratio) for gender, ratio in cohort_ratio.items())
        fairness_total += max(0.0, 1.0 - (distance / 2.0))
        diversity_parts.append(f"{project}: gender={gender_counts}")
    return {
        "preference_counts": counts,
        "average_rank": total_rank / max(1, student_total),
        "fairness_score": 100.0 * fairness_total / max(1, active_groups),
        "fit_count": fit_count,
        "active_groups": active_groups,
        "size_min": min(sizes) if sizes else 0,
        "size_max": max(sizes) if sizes else 0,
        "diversity_summary": "; ".join(diversity_parts),
        "student_total": student_total,
    }


def _ascii_bar(value: float, maximum: float, width: int = 24) -> str:
    if maximum <= 0:
        return "[" + ("-" * width) + "]"
    filled = max(0, min(width, round((value / maximum) * width)))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _parse_allocation_rows(allocations_path: Path, emails_path: Path) -> tuple[List[Dict[str, str]], Dict[str, Dict[str, object]]]:
    email_data = {
        item["project"]: item
        for item in json.loads(emails_path.read_text(encoding="utf-8"))
    }
    groups: Dict[str, Dict[str, object]] = {}
    with allocations_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            project = row["ProjectName"]
            entry = groups.setdefault(
                project,
                {
                    "project": project,
                    "project_id": row["ProjectID"],
                    "description": row["Description"],
                    "difficulty": row["Difficulty"],
                    "size": 0,
                    "range": f"{row['ProjectMinTeamSize']}-{row['ProjectMaxTeamSize']}",
                    "male": 0,
                    "female": 0,
                    "students": [],
                    "to": email_data.get(project, {}).get("to", []),
                    "subject": email_data.get(project, {}).get("subject", f"Your project group assignment: {project}"),
                    "body": email_data.get(project, {}).get("body", ""),
                },
            )
            entry["size"] = int(entry["size"]) + 1
            gender = row["Gender"].strip().lower()
            if gender.startswith("m"):
                entry["male"] = int(entry["male"]) + 1
            elif gender.startswith("f"):
                entry["female"] = int(entry["female"]) + 1
            entry["students"].append(
                {
                    "student_id": row["StudentID"],
                    "name": row["Name"],
                    "gender": row["Gender"],
                    "major": row["Major"],
                    "nationality": row["Nationality"],
                    "email": row["Email"],
                }
            )

    rows = []
    for project in sorted(groups):
        entry = groups[project]
        rows.append(
            {
                "project": str(entry["project"]),
                "size": str(entry["size"]),
                "range": str(entry["range"]),
                "gender": f"F {entry['female']} / M {entry['male']}",
            }
        )
    return rows, groups
