import random
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from puzzle import (
    ANIMATION_DELAY_MS,
    GOAL_STATE,
    PUZZLE_SIZE,
    SCRAMBLE_MOVES,
    apply_move,
    is_solvable,
    puzzle_neighbors,
    solve_puzzle_astar,
    solve_puzzle_backjumping,
    solve_puzzle_bfs,
    solve_puzzle_ida,
)
from utils import SolverStats, StopSearch, format_stats, run_with_stats

ALGORITHM_LABELS = {
    "A* (Манхэттен)": "A* (Manhattan)",
    "Поиск в ширину (BFS)": "BFS",
    "IDA*": "IDA*",
    "Обратные прыжки": "Backjumping",
}


class PuzzleTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.algorithm_var = tk.StringVar(value="A* (Манхэттен)")
        self.status_var = tk.StringVar(value="Готово")
        self.result_var = tk.StringVar(value="")
        self.stop_event = None
        self.worker = None
        self.cell_size = 70
        self.state = GOAL_STATE
        self.rects = []
        self.texts = []

        controls = ttk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(controls, text="Алгоритм").grid(row=0, column=0, sticky=tk.W)
        ttk.OptionMenu(
            controls,
            self.algorithm_var,
            self.algorithm_var.get(),
            *ALGORITHM_LABELS.keys(),
        ).grid(row=0, column=1, padx=5)
        ttk.Button(controls, text="Перемешать", command=self.randomize).grid(
            row=0, column=2, padx=5
        )
        ttk.Button(controls, text="Сбросить", command=self.reset).grid(
            row=0, column=3, padx=5
        )
        ttk.Button(controls, text="Решить", command=self.solve).grid(
            row=0, column=4, padx=5
        )
        ttk.Button(controls, text="Стоп", command=self.stop).grid(
            row=0, column=5, padx=5
        )

        status = ttk.Frame(self)
        status.pack(side=tk.TOP, fill=tk.X, padx=10)
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self, background="#f0f0f0")
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        info = ttk.Frame(self)
        info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        ttk.Label(info, text="Результаты").pack(anchor=tk.W)
        ttk.Label(info, textvariable=self.result_var, justify=tk.LEFT).pack(
            anchor=tk.NW, fill=tk.BOTH, expand=True
        )

        self.build_board()
        self.render()

    def build_board(self):
        self.canvas.delete("all")
        self.rects = [
            [None for _ in range(PUZZLE_SIZE)] for _ in range(PUZZLE_SIZE)
        ]
        self.texts = [
            [None for _ in range(PUZZLE_SIZE)] for _ in range(PUZZLE_SIZE)
        ]
        self.canvas.config(
            width=PUZZLE_SIZE * self.cell_size, height=PUZZLE_SIZE * self.cell_size
        )
        for r in range(PUZZLE_SIZE):
            for c in range(PUZZLE_SIZE):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                rect = self.canvas.create_rectangle(
                    x1, y1, x2, y2, outline="#555", fill="white"
                )
                text = self.canvas.create_text(
                    x1 + self.cell_size / 2,
                    y1 + self.cell_size / 2,
                    text="",
                    font=("Arial", 18, "bold"),
                )
                self.rects[r][c] = rect
                self.texts[r][c] = text

    def render(self):
        for idx, value in enumerate(self.state):
            row, col = divmod(idx, PUZZLE_SIZE)
            if value == 0:
                self.canvas.itemconfig(self.rects[row][col], fill="#d0d0d0")
                self.canvas.itemconfig(self.texts[row][col], text="")
            else:
                self.canvas.itemconfig(self.rects[row][col], fill="white")
                self.canvas.itemconfig(self.texts[row][col], text=str(value))

    def randomize(self):
        self.state = GOAL_STATE
        for _ in range(SCRAMBLE_MOVES):
            neighbors = list(puzzle_neighbors(self.state))
            _, next_state = random.choice(neighbors)
            self.state = next_state
        self.render()
        self.result_var.set("")

    def reset(self):
        self.state = GOAL_STATE
        self.render()
        self.result_var.set("")

    def on_canvas_click(self, event):
        row = event.y // self.cell_size
        col = event.x // self.cell_size
        if not (0 <= row < PUZZLE_SIZE and 0 <= col < PUZZLE_SIZE):
            return
        idx = row * PUZZLE_SIZE + col
        idx0 = self.state.index(0)
        row0, col0 = divmod(idx0, PUZZLE_SIZE)
        if abs(row - row0) + abs(col - col0) == 1:
            new_state = list(self.state)
            new_state[idx0], new_state[idx] = new_state[idx], new_state[idx0]
            self.state = tuple(new_state)
            self.render()

    def set_busy(self, busy):
        self.status_var.set("Выполняется..." if busy else "Готово")

    def solve(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Занято", "Решатель уже работает.")
            return
        if not is_solvable(self.state):
            messagebox.showwarning("Неразрешимо", "Эта позиция неразрешима.")
            return
        self.stop_event = threading.Event()
        algorithm = ALGORITHM_LABELS[self.algorithm_var.get()]
        self.set_busy(True)
        self.result_var.set("")

        def task():
            try:
                if algorithm == "A* (Manhattan)":
                    result, elapsed, peak_kb = run_with_stats(
                        lambda: solve_puzzle_astar(self.state, self.stop_event)
                    )
                    path, nodes = result
                    stats = SolverStats(None, path, nodes, 0, elapsed, peak_kb)
                elif algorithm == "BFS":
                    result, elapsed, peak_kb = run_with_stats(
                        lambda: solve_puzzle_bfs(self.state, self.stop_event)
                    )
                    path, nodes = result
                    stats = SolverStats(None, path, nodes, 0, elapsed, peak_kb)
                elif algorithm == "IDA*":
                    result, elapsed, peak_kb = run_with_stats(
                        lambda: solve_puzzle_ida(self.state, self.stop_event)
                    )
                    path, nodes = result
                    stats = SolverStats(None, path, nodes, 0, elapsed, peak_kb)
                else:
                    result, elapsed, peak_kb = run_with_stats(
                        lambda: solve_puzzle_backjumping(self.state, self.stop_event)
                    )
                    path, nodes, backjumps = result
                    stats = SolverStats(None, path, nodes, backjumps, elapsed, peak_kb)
            except StopSearch:
                stats = None
            self.after(0, lambda: self.finish_task(stats))

        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def finish_task(self, stats):
        self.set_busy(False)
        if stats is None:
            self.result_var.set("Остановлено.")
            return
        if stats.moves is None:
            self.result_var.set("Нет решения.")
            return
        self.result_var.set(format_stats(stats))
        self.animate_solution(stats.moves)

    def animate_solution(self, moves, delay=ANIMATION_DELAY_MS):
        """Animate a list of moves with a delay between frames.

        Args:
            moves: List of move codes ('U', 'D', 'L', 'R') for blank moves
                (e.g., 'U' moves the blank up, decreasing its row index).
            delay: Delay in milliseconds between animation frames.
        """
        if not moves:
            return

        def step(index):
            if index >= len(moves):
                return
            self.state = apply_move(self.state, moves[index])
            self.render()
            self.after(delay, lambda: step(index + 1))

        step(0)

    def stop(self):
        if self.stop_event:
            self.stop_event.set()
