import threading
import tkinter as tk
from tkinter import messagebox, ttk

from hamilton import solve_hamilton
from utils import StopSearch, format_stats, run_with_stats


class HamiltonTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.rows_var = tk.IntVar(value=7)
        self.cols_var = tk.IntVar(value=7)
        self.mode_var = tk.StringVar(value="Set start")
        self.algorithm_var = tk.StringVar(value="Backtracking")
        self.status_var = tk.StringVar(value="Ready")
        self.result_var = tk.StringVar(value="")
        self.stop_event = None
        self.worker = None
        self.cell_size = 35
        self.start = None
        self.finish = None
        self.blocked = set()
        self.rects = []

        controls = ttk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(controls, text="Rows").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(controls, from_=2, to=10, textvariable=self.rows_var, width=5).grid(
            row=0, column=1, sticky=tk.W
        )
        ttk.Label(controls, text="Cols").grid(row=0, column=2, sticky=tk.W)
        ttk.Spinbox(controls, from_=2, to=10, textvariable=self.cols_var, width=5).grid(
            row=0, column=3, sticky=tk.W
        )
        ttk.Button(controls, text="Apply size", command=self.build_grid).grid(
            row=0, column=4, padx=5
        )
        ttk.Label(controls, text="Mode").grid(row=0, column=5, sticky=tk.W)
        ttk.OptionMenu(
            controls,
            self.mode_var,
            self.mode_var.get(),
            "Set start",
            "Set finish",
            "Toggle block",
        ).grid(row=0, column=6, padx=5)
        ttk.Label(controls, text="Algorithm").grid(row=0, column=7, sticky=tk.W)
        ttk.OptionMenu(
            controls,
            self.algorithm_var,
            self.algorithm_var.get(),
            "Backtracking",
            "Warnsdorff",
            "Connectivity pruning",
            "Backjumping",
        ).grid(row=0, column=8, padx=5)
        ttk.Button(controls, text="Solve", command=self.solve).grid(
            row=0, column=9, padx=5
        )
        ttk.Button(controls, text="Stop", command=self.stop).grid(
            row=0, column=10, padx=5
        )
        ttk.Button(controls, text="Clear", command=self.clear_grid).grid(
            row=0, column=11, padx=5
        )

        status = ttk.Frame(self)
        status.pack(side=tk.TOP, fill=tk.X, padx=10)
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self, background="white")
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        info = ttk.Frame(self)
        info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        ttk.Label(info, text="Results").pack(anchor=tk.W)
        ttk.Label(info, textvariable=self.result_var, justify=tk.LEFT).pack(
            anchor=tk.NW, fill=tk.BOTH, expand=True
        )

        self.build_grid()

    def build_grid(self):
        self.canvas.delete("all")
        self.blocked.clear()
        self.start = None
        self.finish = None
        rows = self.rows_var.get()
        cols = self.cols_var.get()
        self.rects = [[None for _ in range(cols)] for _ in range(rows)]
        self.canvas.config(width=cols * self.cell_size, height=rows * self.cell_size)
        for r in range(rows):
            for c in range(cols):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                rect = self.canvas.create_rectangle(
                    x1, y1, x2, y2, outline="#888", fill="white"
                )
                self.rects[r][c] = rect
        self.result_var.set("")

    def clear_grid(self):
        self.build_grid()

    def set_cell(self, pos, color):
        row, col = pos
        self.canvas.itemconfig(self.rects[row][col], fill=color)

    def on_canvas_click(self, event):
        row = event.y // self.cell_size
        col = event.x // self.cell_size
        rows = self.rows_var.get()
        cols = self.cols_var.get()
        if not (0 <= row < rows and 0 <= col < cols):
            return
        pos = (row, col)
        mode = self.mode_var.get()
        if mode == "Set start":
            if self.start:
                self.set_cell(self.start, "white")
            if pos == self.finish:
                self.finish = None
            self.start = pos
            self.blocked.discard(pos)
            self.set_cell(pos, "green")
        elif mode == "Set finish":
            if self.finish:
                self.set_cell(self.finish, "white")
            if pos == self.start:
                self.start = None
            self.finish = pos
            self.blocked.discard(pos)
            self.set_cell(pos, "red")
        elif mode == "Toggle block":
            if pos in self.blocked:
                self.blocked.remove(pos)
                self.set_cell(pos, "white")
            else:
                if pos == self.start:
                    self.start = None
                if pos == self.finish:
                    self.finish = None
                self.blocked.add(pos)
                self.set_cell(pos, "#444")

    def set_busy(self, busy):
        self.status_var.set("Working..." if busy else "Ready")

    def solve(self):
        if not self.start or not self.finish:
            messagebox.showwarning("Input", "Select start and finish cells.")
            return
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "Solver is already running.")
            return
        self.stop_event = threading.Event()
        rows = self.rows_var.get()
        cols = self.cols_var.get()
        mode = self.algorithm_var.get()
        self.set_busy(True)
        self.result_var.set("")

        def task():
            try:
                result, elapsed, peak_kb = run_with_stats(
                    lambda: solve_hamilton(
                        rows,
                        cols,
                        self.start,
                        self.finish,
                        set(self.blocked),
                        mode,
                        self.stop_event,
                    )
                )
            except StopSearch:
                result = None
                elapsed = 0.0
                peak_kb = 0.0
            self.after(0, lambda: self.finish_task(result, elapsed, peak_kb))

        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def finish_task(self, result, elapsed, peak_kb):
        self.set_busy(False)
        if result is None:
            self.result_var.set("Stopped.")
            return
        result.elapsed = elapsed
        result.peak_kb = peak_kb
        self.result_var.set(format_stats(result))

    def stop(self):
        if self.stop_event:
            self.stop_event.set()
