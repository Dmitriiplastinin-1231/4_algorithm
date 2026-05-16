import argparse
import csv
import heapq
import os
import random
import time
from collections import Counter
from statistics import mean

import puzzle
from puzzle import INF, MAX_BACKJUMP_DEPTH, puzzle_neighbors, set_puzzle_size
from research_hamilton import parse_pos, solve_hamilton_backjump, solve_hamilton_dfs


TASK1_OUTPUT = "task1_algorithm_comparison.csv"
TASK2_OUTPUT = "task2_algorithm_comparison.csv"


def ensure_output_dir(path: str) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    ensure_output_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def avg(values: list[float | int | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return mean(filtered)


def format_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_table(title: str, rows: list[dict], columns: list[str]) -> None:
    if not rows:
        print(f"{title}\n(no data)\n")
        return

    widths = {}
    for column in columns:
        widths[column] = max(len(column), *(len(format_value(row.get(column))) for row in rows))

    print(title)
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    print(header)
    print(separator)
    for row in rows:
        print(" | ".join(format_value(row.get(column)).ljust(widths[column]) for column in columns))
    print()


def summarize(rows: list[dict], group_key: str, metrics: list[str]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), []).append(row)

    summary_rows = []
    for group_name, group_rows in groups.items():
        status_counts = Counter(str(row["status"]) for row in group_rows)
        summary = {
            group_key: group_name,
            "params": group_rows[0]["params"],
            "trials": len(group_rows),
            "ok_runs": status_counts.get("ok", 0),
            "statuses": ", ".join(f"{status}:{count}" for status, count in sorted(status_counts.items())),
        }
        for metric in metrics:
            summary[f"avg_{metric}"] = avg([row.get(metric) for row in group_rows])
        summary_rows.append(summary)

    return summary_rows


def random_scramble(depth: int, rng: random.Random) -> tuple[int, ...]:
    state = puzzle.GOAL_STATE
    prev_move = None
    opposite = {"U": "D", "D": "U", "L": "R", "R": "L"}
    for _ in range(depth):
        moves = list(puzzle_neighbors(state))
        if prev_move is not None:
            filtered = [move for move in moves if move[0] != opposite[prev_move]]
            if filtered:
                moves = filtered
        move, next_state = rng.choice(moves)
        state = next_state
        prev_move = move
    return state


def manhattan_distance(state: tuple[int, ...]) -> int:
    return puzzle.manhattan(state)


def linear_conflict_score(state: tuple[int, ...]) -> int:
    size = puzzle.PUZZLE_SIZE
    conflicts = 0

    for row in range(size):
        row_tiles = []
        for col in range(size):
            value = state[row * size + col]
            if value == 0:
                continue
            goal_row, goal_col = puzzle.GOAL_POS[value]
            if goal_row == row:
                row_tiles.append(goal_col)
        for i in range(len(row_tiles)):
            for j in range(i + 1, len(row_tiles)):
                if row_tiles[i] > row_tiles[j]:
                    conflicts += 1

    for col in range(size):
        col_tiles = []
        for row in range(size):
            value = state[row * size + col]
            if value == 0:
                continue
            goal_row, goal_col = puzzle.GOAL_POS[value]
            if goal_col == col:
                col_tiles.append(goal_row)
        for i in range(len(col_tiles)):
            for j in range(i + 1, len(col_tiles)):
                if col_tiles[i] > col_tiles[j]:
                    conflicts += 1

    return conflicts


def ida_star_limited(
    start: tuple[int, ...],
    heuristic,
    *,
    time_limit_sec: float,
    node_limit: int,
) -> tuple[list[str] | None, int, str]:
    bound = heuristic(start)
    path: list[str] = []
    visited = {start}
    expanded = 0
    t0 = time.perf_counter()

    def dfs(state: tuple[int, ...], depth: int, bound_value: float):
        nonlocal expanded
        if expanded >= node_limit:
            return "NODE_LIMIT"
        if (time.perf_counter() - t0) >= time_limit_sec:
            return "TIME_LIMIT"
        expanded += 1

        score = depth + heuristic(state)
        if score > bound_value:
            return score
        if state == puzzle.GOAL_STATE:
            return "FOUND"

        minimum = INF
        for move, next_state in puzzle_neighbors(state):
            if next_state in visited:
                continue
            visited.add(next_state)
            path.append(move)
            result = dfs(next_state, depth + 1, bound_value)
            if result == "FOUND":
                return "FOUND"
            if result in ("TIME_LIMIT", "NODE_LIMIT"):
                return result
            if result < minimum:
                minimum = result
            path.pop()
            visited.remove(next_state)
        return minimum

    while True:
        result = dfs(start, 0, bound)
        if result == "FOUND":
            return list(path), expanded, "ok"
        if result == "TIME_LIMIT":
            return None, expanded, "time_limit"
        if result == "NODE_LIMIT":
            return None, expanded, "node_limit"
        if result == INF:
            return None, expanded, "no_solution"
        bound = result


def bfs_limited(
    start: tuple[int, ...],
    *,
    time_limit_sec: float,
    node_limit: int,
) -> tuple[list[str] | None, int, int, str]:
    t0 = time.perf_counter()
    queue = [start]
    head = 0
    parent = {start: (None, None)}
    expanded = 0
    peak_open_size = 1

    while head < len(queue):
        if expanded >= node_limit:
            return None, expanded, peak_open_size, "node_limit"
        if (time.perf_counter() - t0) >= time_limit_sec:
            return None, expanded, peak_open_size, "time_limit"

        state = queue[head]
        head += 1
        expanded += 1
        if state == puzzle.GOAL_STATE:
            return puzzle.build_path(parent, state), expanded, peak_open_size, "ok"

        for move, next_state in puzzle_neighbors(state):
            if next_state not in parent:
                parent[next_state] = (state, move)
                queue.append(next_state)
        peak_open_size = max(peak_open_size, len(queue) - head)

    return None, expanded, peak_open_size, "no_solution"


def astar_limited(start: tuple[int, ...], *, time_limit_sec: float, node_limit: int) -> tuple[list[str] | None, int, int, str]:
    open_heap: list[tuple[int, int, tuple[int, ...]]] = []
    g_score = {start: 0}
    came_from = {start: (None, None)}
    heapq.heappush(open_heap, (manhattan_distance(start), 0, start))
    expanded = 0
    peak_open_size = 1
    t0 = time.perf_counter()

    while open_heap:
        if expanded >= node_limit:
            return None, expanded, peak_open_size, "node_limit"
        if (time.perf_counter() - t0) >= time_limit_sec:
            return None, expanded, peak_open_size, "time_limit"

        _, g, state = heapq.heappop(open_heap)
        if g != g_score.get(state):
            continue

        expanded += 1
        if state == puzzle.GOAL_STATE:
            return puzzle.build_path(came_from, state), expanded, peak_open_size, "ok"

        for move, nxt in puzzle_neighbors(state):
            next_g = g + 1
            if next_g < g_score.get(nxt, INF):
                g_score[nxt] = next_g
                came_from[nxt] = (state, move)
                heapq.heappush(open_heap, (next_g + manhattan_distance(nxt), next_g, nxt))
        peak_open_size = max(peak_open_size, len(open_heap))

    return None, expanded, peak_open_size, "no_solution"


def backjumping_limited(
    start: tuple[int, ...],
    *,
    time_limit_sec: float,
    node_limit: int,
) -> tuple[list[str] | None, int, int, str]:
    depth = 0
    nodes_total = 0
    backjumps = 0
    t0 = time.perf_counter()

    def stop_status() -> str | None:
        if nodes_total >= node_limit:
            return "node_limit"
        if (time.perf_counter() - t0) >= time_limit_sec:
            return "time_limit"
        return None

    def ordered_neighbors(state: tuple[int, ...]) -> list[tuple[str, tuple[int, ...]]]:
        moves = list(puzzle_neighbors(state))
        moves.sort(key=lambda item: manhattan_distance(item[1]))
        return moves

    def dfs(state: tuple[int, ...], depth_limit: int, visited: set[tuple[int, ...]], path: list[str]):
        nonlocal nodes_total, backjumps
        status = stop_status()
        if status is not None:
            return status
        nodes_total += 1
        if state == puzzle.GOAL_STATE:
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
            result = dfs(nxt, depth_limit - 1, visited, path)
            if result is True:
                return True
            if result in ("time_limit", "node_limit"):
                return result
            path.pop()
            visited.remove(nxt)

        if not any_branch and depth_limit > 0:
            backjumps += 1
        return False

    while depth <= MAX_BACKJUMP_DEPTH:
        status = stop_status()
        if status is not None:
            return None, nodes_total, backjumps, status
        path: list[str] = []
        visited = {start}
        result = dfs(start, depth, visited, path)
        if result is True:
            return path, nodes_total, backjumps, "ok"
        if result in ("time_limit", "node_limit"):
            return None, nodes_total, backjumps, result
        depth += 1

    return None, nodes_total, backjumps, "depth_limit"


def run_task1(args: argparse.Namespace) -> list[dict]:
    start = parse_pos(args.hamilton_start)
    finish = parse_pos(args.hamilton_finish)

    specs = [
        {
            "algorithm": "Backtracking",
            "params": "baseline",
            "runner": lambda trial: solve_hamilton_dfs(
                args.hamilton_rows,
                args.hamilton_cols,
                start,
                finish,
                warnsdorff_alpha=0.0,
                connectivity_policy="never",
                rng=random.Random(args.seed + trial),
                time_limit_sec=args.time_limit,
                node_limit=args.node_limit,
            ),
        },
        {
            "algorithm": "Warnsdorff",
            "params": "alpha=0.1",
            "runner": lambda trial: solve_hamilton_dfs(
                args.hamilton_rows,
                args.hamilton_cols,
                start,
                finish,
                warnsdorff_alpha=0.1,
                connectivity_policy="never",
                rng=random.Random(args.seed + 100 + trial),
                time_limit_sec=args.time_limit,
                node_limit=args.node_limit,
            ),
        },
        {
            "algorithm": "Connectivity pruning",
            "params": "policy=1",
            "runner": lambda trial: solve_hamilton_dfs(
                args.hamilton_rows,
                args.hamilton_cols,
                start,
                finish,
                warnsdorff_alpha=0.0,
                connectivity_policy="1",
                rng=random.Random(args.seed + 200 + trial),
                time_limit_sec=args.time_limit,
                node_limit=args.node_limit,
            ),
        },
        {
            "algorithm": "Backjumping",
            "params": "mode=max_conflict",
            "runner": lambda trial: solve_hamilton_backjump(
                args.hamilton_rows,
                args.hamilton_cols,
                start,
                finish,
                mode="max_conflict",
                warnsdorff_alpha=0.0,
                rng=random.Random(args.seed + 300 + trial),
                time_limit_sec=args.time_limit,
                node_limit=args.node_limit,
            ),
        },
    ]

    rows = []
    for trial in range(args.hamilton_trials):
        for spec in specs:
            metrics = spec["runner"](trial)
            rows.append(
                {
                    "algorithm": spec["algorithm"],
                    "params": spec["params"],
                    "trial": trial,
                    "elapsed": metrics.elapsed,
                    "recursive_calls": metrics.recursive_calls,
                    "backtracks": metrics.backtracks,
                    "dead_ends": metrics.dead_ends,
                    "solutions": metrics.solutions,
                    "pruned_states": metrics.pruned_states,
                    "jumps": metrics.jumps,
                    "depth_reduction": metrics.depth_reduction,
                    "max_depth": metrics.max_depth,
                    "status": metrics.status,
                }
            )

    summary_rows = summarize(
        rows,
        "algorithm",
        [
            "elapsed",
            "recursive_calls",
            "backtracks",
            "dead_ends",
            "solutions",
            "pruned_states",
            "jumps",
            "depth_reduction",
            "max_depth",
        ],
    )

    print_table(
        "Task 1 algorithm comparison",
        summary_rows,
        [
            "algorithm",
            "params",
            "trials",
            "ok_runs",
            "avg_elapsed",
            "avg_recursive_calls",
            "avg_backtracks",
            "avg_dead_ends",
            "avg_solutions",
            "avg_pruned_states",
            "avg_jumps",
            "avg_depth_reduction",
            "avg_max_depth",
            "statuses",
        ],
    )
    return summary_rows


def run_task2(args: argparse.Namespace) -> list[dict]:
    set_puzzle_size(args.puzzle_size)
    rng = random.Random(args.seed)
    rows = []

    def measure_astar(start_state: tuple[int, ...]) -> dict:
        t0 = time.perf_counter()
        path, expanded, peak_open, status = astar_limited(
            start_state,
            time_limit_sec=args.time_limit,
            node_limit=args.node_limit,
        )
        return {
            "path": path,
            "expanded": expanded,
            "peak_open_size": peak_open,
            "status": status,
            "elapsed": time.perf_counter() - t0,
            "backjumps": 0,
        }

    def measure_bfs(start_state: tuple[int, ...]) -> dict:
        t0 = time.perf_counter()
        path, expanded, peak_open, status = bfs_limited(
            start_state,
            time_limit_sec=args.time_limit,
            node_limit=args.node_limit,
        )
        return {
            "path": path,
            "expanded": expanded,
            "peak_open_size": peak_open,
            "status": status,
            "elapsed": time.perf_counter() - t0,
            "backjumps": 0,
        }

    def measure_ida(start_state: tuple[int, ...], heuristic) -> dict:
        t0 = time.perf_counter()
        path, expanded, status = ida_star_limited(
            start_state,
            heuristic=heuristic,
            time_limit_sec=args.time_limit,
            node_limit=args.node_limit,
        )
        return {
            "path": path,
            "expanded": expanded,
            "peak_open_size": None,
            "status": status,
            "elapsed": time.perf_counter() - t0,
            "backjumps": 0,
        }

    def measure_backjumping(start_state: tuple[int, ...]) -> dict:
        t0 = time.perf_counter()
        path, expanded, backjumps, status = backjumping_limited(
            start_state,
            time_limit_sec=args.time_limit,
            node_limit=args.node_limit,
        )
        return {
            "path": path,
            "expanded": expanded,
            "peak_open_size": None,
            "status": status,
            "elapsed": time.perf_counter() - t0,
            "backjumps": backjumps,
        }

    for trial in range(args.puzzle_trials):
        start = random_scramble(args.puzzle_scramble_depth, rng)

        astar_result = measure_astar(start)
        astar_path = astar_result["path"]
        optimal_length = len(astar_path) if astar_path is not None and astar_result["status"] == "ok" else None

        task2_runs = [
            (
                "A* (Manhattan)",
                "baseline",
                lambda: astar_result,
            ),
            (
                "BFS",
                "baseline",
                lambda: measure_bfs(start),
            ),
            (
                "IDA*",
                "w=1.0",
                lambda: measure_ida(start, manhattan_distance),
            ),
            (
                "IDA* + Linear conflict",
                "beta=2",
                lambda: measure_ida(
                    start,
                    lambda state: manhattan_distance(state) + 2 * linear_conflict_score(state),
                ),
            ),
            (
                "Backjumping",
                "baseline",
                lambda: measure_backjumping(start),
            ),
        ]

        for algorithm, params, runner in task2_runs:
            result = runner()
            path = result["path"]
            expanded = result["expanded"]
            peak_open = result["peak_open_size"]
            status = result["status"]
            elapsed = result["elapsed"]
            backjumps = result["backjumps"]
            solution_length = len(path) if path is not None else None
            optimality = None
            if solution_length is not None and optimal_length is not None:
                if optimal_length == 0:
                    optimality = 1.0 if solution_length == 0 else None
                else:
                    optimality = solution_length / optimal_length
            elif solution_length == 0 and optimal_length == 0:
                optimality = 1.0

            rows.append(
                {
                    "algorithm": algorithm,
                    "params": params,
                    "trial": trial,
                    "elapsed": elapsed,
                    "expanded": expanded,
                    "solution_length": solution_length,
                    "optimal_length": optimal_length,
                    "optimality": optimality,
                    "peak_open_size": peak_open,
                    "backjumps": backjumps,
                    "status": status,
                }
            )

    summary_rows = summarize(
        rows,
        "algorithm",
        [
            "elapsed",
            "expanded",
            "solution_length",
            "optimal_length",
            "optimality",
            "peak_open_size",
            "backjumps",
        ],
    )

    print_table(
        "Task 2 algorithm comparison",
        summary_rows,
        [
            "algorithm",
            "params",
            "trials",
            "ok_runs",
            "avg_elapsed",
            "avg_expanded",
            "avg_solution_length",
            "avg_optimal_length",
            "avg_optimality",
            "avg_peak_open_size",
            "avg_backjumps",
            "statuses",
        ],
    )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the best algorithm variants for both tasks")
    parser.add_argument("--task", choices=["task1", "task2", "all"], default="all")
    parser.add_argument("--output-dir", default="comparison_outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--node-limit", type=int, default=500000)
    parser.add_argument("--hamilton-rows", type=int, default=4)
    parser.add_argument("--hamilton-cols", type=int, default=4)
    parser.add_argument("--hamilton-start", type=str, default="0,0")
    parser.add_argument("--hamilton-finish", type=str, default="3,3")
    parser.add_argument("--hamilton-trials", type=int, default=3)
    parser.add_argument("--puzzle-size", type=int, default=4)
    parser.add_argument("--puzzle-scramble-depth", type=int, default=20)
    parser.add_argument("--puzzle-trials", type=int, default=5)
    args = parser.parse_args()

    if args.task in ("task1", "all"):
        task1_rows = run_task1(args)
        write_csv(os.path.join(args.output_dir, TASK1_OUTPUT), task1_rows)

    if args.task in ("task2", "all"):
        task2_rows = run_task2(args)
        write_csv(os.path.join(args.output_dir, TASK2_OUTPUT), task2_rows)


if __name__ == "__main__":
    main()
