import heapq
import random
import threading
import time
import tracemalloc
from collections import deque
from dataclasses import dataclass
import tkinter as tk
from tkinter import messagebox, ttk


class StopSearch(Exception):
    pass


@dataclass
class SolverStats:
    solutions: int | None
    moves: list[str] | None
    nodes: int
    backjumps: int
    elapsed: float
    peak_kb: float


GOAL_STATE = tuple(list(range(1, 16)) + [0])
GOAL_POS = {value: divmod(idx, 4) for idx, value in enumerate(GOAL_STATE)}
INF = float("inf")
RANDOMIZE_MOVES = 120


def run_with_stats(func):
    tracemalloc.start()
    start = time.perf_counter()
    try:
        result = func()
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    elapsed = time.perf_counter() - start
    return result, elapsed, peak / 1024


def format_stats(stats: SolverStats) -> str:
    lines = []
    if stats.solutions is not None:
        lines.append(f"Paths: {stats.solutions}")
    if stats.moves is not None:
        lines.append(f"Moves: {len(stats.moves)}")
    lines.append(f"Nodes expanded: {stats.nodes}")
    if stats.backjumps:
        lines.append(f"Backjumps: {stats.backjumps}")
    lines.append(f"Time: {stats.elapsed:.3f}s")
    lines.append(f"Peak memory: {stats.peak_kb:.1f} KB")
    return "\n".join(lines)


# Hamiltonian path solvers

def grid_neighbors(pos, rows, cols):
    row, col = pos
    if row > 0:
        yield row - 1, col
    if row + 1 < rows:
        yield row + 1, col
    if col > 0:
        yield row, col - 1
    if col + 1 < cols:
        yield row, col + 1


def connectivity_ok(free_cells, visited, rows, cols):
    unvisited = free_cells - visited
    if not unvisited:
        return True
    start = next(iter(unvisited))
    queue = deque([start])
    seen = {start}
    while queue:
        pos = queue.popleft()
        for neighbor in grid_neighbors(pos, rows, cols):
            if neighbor in unvisited and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen) == len(unvisited)


def unvisited_degree(pos, free_cells, visited, rows, cols):
    count = 0
    for neighbor in grid_neighbors(pos, rows, cols):
        if neighbor in free_cells and neighbor not in visited:
            count += 1
    return count


def solve_hamilton(rows, cols, start, finish, blocked, mode, stop_event=None):
    free_cells = {
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (r, c) not in blocked
    }
    if start not in free_cells or finish not in free_cells:
        return SolverStats(0, None, 0, 0, 0.0, 0.0)

    free_count = len(free_cells)
    visited = {start}
    path = [start]
    solutions = 0
    nodes = 0

    use_warnsdorff = mode == "Warnsdorff"
    use_connectivity = mode == "Connectivity pruning"

    def ordered_moves(pos):
        moves = [
            n
            for n in grid_neighbors(pos, rows, cols)
            if n in free_cells and n not in visited
        ]
        if use_warnsdorff:
            moves.sort(key=lambda n: unvisited_degree(n, free_cells, visited, rows, cols))
        return moves

    def backtrack(pos):
        nonlocal solutions, nodes
        if stop_event and stop_event.is_set():
            raise StopSearch()
        nodes += 1
        if pos == finish and len(path) != free_count:
            return
        if len(path) == free_count:
            if pos == finish:
                solutions += 1
            return
        if use_connectivity and not connectivity_ok(free_cells, visited, rows, cols):
            return
        for nxt in ordered_moves(pos):
            if nxt == finish and len(path) != free_count - 1:
                continue
            visited.add(nxt)
            path.append(nxt)
            backtrack(nxt)
            path.pop()
            visited.remove(nxt)

    def backjumping():
        nonlocal solutions, nodes
        backjumps = 0
        stack = [{"pos": start, "moves": ordered_moves(start)}]
        while stack:
            if stop_event and stop_event.is_set():
                raise StopSearch()
            frame = stack[-1]
            pos = frame["pos"]
            if len(path) == free_count:
                if pos == finish:
                    solutions += 1
                stack.pop()
                visited.remove(pos)
                path.pop()
                continue
            if not frame["moves"]:
                stack.pop()
                visited.remove(pos)
                path.pop()
                jump_count = 0
                while stack and not stack[-1]["moves"]:
                    frame = stack.pop()
                    visited.remove(frame["pos"])
                    path.pop()
                    jump_count += 1
                if jump_count:
                    backjumps += jump_count
                continue
            nxt = frame["moves"].pop(0)
            if nxt == finish and len(path) != free_count - 1:
                continue
            visited.add(nxt)
            path.append(nxt)
            nodes += 1
            stack.append({"pos": nxt, "moves": ordered_moves(nxt)})
        return backjumps

    backjumps = 0
    if mode == "Backjumping":
        try:
            backjumps = backjumping()
        except StopSearch:
            return None
    else:
        try:
            backtrack(start)
        except StopSearch:
            return None
    return SolverStats(solutions, None, nodes, backjumps, 0.0, 0.0)


# 15-puzzle helpers

def manhattan(state):
    total = 0
    for idx, value in enumerate(state):
        if value == 0:
            continue
        row, col = divmod(idx, 4)
        goal_row, goal_col = GOAL_POS[value]
        total += abs(row - goal_row) + abs(col - goal_col)
    return total


def puzzle_neighbors(state):
    idx0 = state.index(0)
    row, col = divmod(idx0, 4)
    moves = []
    if row > 0:
        moves.append(("U", idx0 - 4))
    if row < 3:
        moves.append(("D", idx0 + 4))
    if col > 0:
        moves.append(("L", idx0 - 1))
    if col < 3:
        moves.append(("R", idx0 + 1))
    for move, idx in moves:
        new_state = list(state)
        new_state[idx0], new_state[idx] = new_state[idx], new_state[idx0]
        yield move, tuple(new_state)


def apply_move(state, move):
    idx0 = state.index(0)
    row, col = divmod(idx0, 4)
    if move == "U":
        idx = idx0 - 4
    elif move == "D":
        idx = idx0 + 4
    elif move == "L":
        idx = idx0 - 1
    elif move == "R":
        idx = idx0 + 1
    else:
        raise ValueError("Unknown move")
    new_state = list(state)
    new_state[idx0], new_state[idx] = new_state[idx], new_state[idx0]
    return tuple(new_state)


def is_solvable(state):
    if len(state) != 16:
        raise ValueError("Expected a 4x4 puzzle state (16 tiles).")
    width = int(len(state) ** 0.5)
    values = [v for v in state if v != 0]
    inversions = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[i] > values[j]:
                inversions += 1
    blank_row_from_bottom = width - (state.index(0) // width)
    if width % 2 == 1:
        return inversions % 2 == 0
    return (blank_row_from_bottom % 2 == 0) != (inversions % 2 == 0)


def solve_puzzle_astar(start, stop_event=None):
    open_heap = []
    g_score = {start: 0}
    came_from = {start: (None, None)}
    heapq.heappush(open_heap, (manhattan(start), 0, start))
    nodes = 0

    while open_heap:
        if stop_event and stop_event.is_set():
            raise StopSearch()
        _, g, state = heapq.heappop(open_heap)
        if g != g_score.get(state):
            continue
        nodes += 1
        if state == GOAL_STATE:
            return build_path(came_from, state), nodes
        for move, nxt in puzzle_neighbors(state):
            ng = g + 1
            if ng < g_score.get(nxt, INF):
                g_score[nxt] = ng
                came_from[nxt] = (state, move)
                heapq.heappush(open_heap, (ng + manhattan(nxt), ng, nxt))
    return None, nodes


def solve_puzzle_bfs(start, stop_event=None):
    queue = deque([start])
    came_from = {start: (None, None)}
    nodes = 0
    while queue:
        if stop_event and stop_event.is_set():
            raise StopSearch()
        state = queue.popleft()
        nodes += 1
        if state == GOAL_STATE:
            return build_path(came_from, state), nodes
        for move, nxt in puzzle_neighbors(state):
            if nxt not in came_from:
                came_from[nxt] = (state, move)
                queue.append(nxt)
    return None, nodes


def solve_puzzle_ida(start, stop_event=None):
    bound = manhattan(start)
    path = []
    visited = {start}
    nodes = 0

    def dfs(state, g, bound):
        nonlocal nodes
        if stop_event and stop_event.is_set():
            raise StopSearch()
        nodes += 1
        f = g + manhattan(state)
        if f > bound:
            return f
        if state == GOAL_STATE:
            return "FOUND"
        minimum = INF
        for move, nxt in puzzle_neighbors(state):
            if nxt in visited:
                continue
            visited.add(nxt)
            path.append(move)
            result = dfs(nxt, g + 1, bound)
            if result == "FOUND":
                return "FOUND"
            if result < minimum:
                minimum = result
            path.pop()
            visited.remove(nxt)
        return minimum

    while True:
        if stop_event and stop_event.is_set():
            raise StopSearch()
        result = dfs(start, 0, bound)
        if result == "FOUND":
            return list(path), nodes
        if result == INF:
            return None, nodes
        bound = result


def solve_puzzle_backjumping(start, stop_event=None):
    depth = 0
    nodes_total = 0
    backjumps = 0

    def ordered_neighbors(state):
        moves = list(puzzle_neighbors(state))
        moves.sort(key=lambda item: manhattan(item[1]))
        return moves

    def dfs(state, depth_limit, visited, path):
        nonlocal nodes_total, backjumps
        if stop_event and stop_event.is_set():
            raise StopSearch()
        nodes_total += 1
        if state == GOAL_STATE:
            return True
        if depth_limit == 0:
            return False
        any_branch = False
        for move, nxt in ordered_neighbors(state):
            if nxt in visited:
                continue
            any_branch = True
            visited.add(nxt)
            path.append(move)
            if dfs(nxt, depth_limit - 1, visited, path):
                return True
            path.pop()
            visited.remove(nxt)
        if not any_branch and depth_limit > 0:
            backjumps += 1
        return False

    while True:
        if stop_event and stop_event.is_set():
            raise StopSearch()
        path = []
        visited = {start}
        found = dfs(start, depth, visited, path)
        if found:
            return path, nodes_total, backjumps
        depth += 1


def build_path(came_from, state):
    moves = []
    while True:
        parent, move = came_from[state]
        if parent is None:
            break
        moves.append(move)
        state = parent
    return list(reversed(moves))


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


class PuzzleTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.algorithm_var = tk.StringVar(value="A* (Manhattan)")
        self.status_var = tk.StringVar(value="Ready")
        self.result_var = tk.StringVar(value="")
        self.stop_event = None
        self.worker = None
        self.cell_size = 70
        self.state = GOAL_STATE
        self.rects = []
        self.texts = []

        controls = ttk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(controls, text="Algorithm").grid(row=0, column=0, sticky=tk.W)
        ttk.OptionMenu(
            controls,
            self.algorithm_var,
            self.algorithm_var.get(),
            "A* (Manhattan)",
            "BFS",
            "IDA*",
            "Backjumping",
        ).grid(row=0, column=1, padx=5)
        ttk.Button(controls, text="Randomize", command=self.randomize).grid(
            row=0, column=2, padx=5
        )
        ttk.Button(controls, text="Reset", command=self.reset).grid(
            row=0, column=3, padx=5
        )
        ttk.Button(controls, text="Solve", command=self.solve).grid(
            row=0, column=4, padx=5
        )
        ttk.Button(controls, text="Stop", command=self.stop).grid(
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
        ttk.Label(info, text="Results").pack(anchor=tk.W)
        ttk.Label(info, textvariable=self.result_var, justify=tk.LEFT).pack(
            anchor=tk.NW, fill=tk.BOTH, expand=True
        )

        self.build_board()
        self.render()

    def build_board(self):
        self.canvas.delete("all")
        self.rects = [[None for _ in range(4)] for _ in range(4)]
        self.texts = [[None for _ in range(4)] for _ in range(4)]
        self.canvas.config(width=4 * self.cell_size, height=4 * self.cell_size)
        for r in range(4):
            for c in range(4):
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
            row, col = divmod(idx, 4)
            if value == 0:
                self.canvas.itemconfig(self.rects[row][col], fill="#d0d0d0")
                self.canvas.itemconfig(self.texts[row][col], text="")
            else:
                self.canvas.itemconfig(self.rects[row][col], fill="white")
                self.canvas.itemconfig(self.texts[row][col], text=str(value))

    def randomize(self):
        self.state = GOAL_STATE
        for _ in range(RANDOMIZE_MOVES):
            _, next_state = random.choice(list(puzzle_neighbors(self.state)))
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
        if not (0 <= row < 4 and 0 <= col < 4):
            return
        idx = row * 4 + col
        idx0 = self.state.index(0)
        row0, col0 = divmod(idx0, 4)
        if abs(row - row0) + abs(col - col0) == 1:
            new_state = list(self.state)
            new_state[idx0], new_state[idx] = new_state[idx], new_state[idx0]
            self.state = tuple(new_state)
            self.render()

    def set_busy(self, busy):
        self.status_var.set("Working..." if busy else "Ready")

    def solve(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "Solver is already running.")
            return
        if not is_solvable(self.state):
            messagebox.showwarning("Unsolvable", "This puzzle position is unsolvable.")
            return
        self.stop_event = threading.Event()
        algorithm = self.algorithm_var.get()
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
            self.result_var.set("Stopped.")
            return
        if stats.moves is None:
            self.result_var.set("No solution.")
            return
        self.result_var.set(format_stats(stats))
        self.animate_solution(stats.moves)

    def animate_solution(self, moves, delay=120):
        """Animate a list of moves with a delay between frames.

        Args:
            moves: List of move codes ('U', 'D', 'L', 'R') for the blank tile.
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


class LabApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lab 4: Optimization Algorithms")
        self.geometry("1100x750")
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)
        notebook.add(HamiltonTab(notebook), text="Hamilton Paths")
        notebook.add(PuzzleTab(notebook), text="15-Puzzle")


if __name__ == "__main__":
    app = LabApp()
    app.mainloop()
