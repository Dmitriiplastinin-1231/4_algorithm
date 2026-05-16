import argparse
import csv
import random
import time
from dataclasses import dataclass

import puzzle
from puzzle import GOAL_STATE, apply_move, puzzle_neighbors, set_puzzle_size


INF = float("inf")


@dataclass
class PuzzleMetrics:
    elapsed: float
    expanded: int
    solution_length: int | None
    optimality: float | None
    peak_open_size: int
    status: str


def random_scramble(depth: int, rng: random.Random) -> tuple[int, ...]:
    state = GOAL_STATE
    prev_move = None
    opposite = {"U": "D", "D": "U", "L": "R", "R": "L"}
    for _ in range(depth):
        moves = list(puzzle_neighbors(state))
        if prev_move is not None:
            filtered = [m for m in moves if m[0] != opposite[prev_move]]
            if filtered:
                moves = filtered
        mv, nxt = rng.choice(moves)
        state = nxt
        prev_move = mv
    return state


def manhattan(state: tuple[int, ...]) -> int:
    return puzzle.manhattan(state)


def linear_conflicts(state: tuple[int, ...]) -> int:
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


def ida_star(
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

    def dfs(state: tuple[int, ...], g: int, bound_val: float):
        nonlocal expanded
        if expanded >= node_limit:
            return "NODE_LIMIT"
        if (time.perf_counter() - t0) >= time_limit_sec:
            return "TIME_LIMIT"
        expanded += 1

        f = g + heuristic(state)
        if f > bound_val:
            return f
        if state == GOAL_STATE:
            return "FOUND"

        minimum = INF
        for move, nxt in puzzle_neighbors(state):
            if nxt in visited:
                continue
            visited.add(nxt)
            path.append(move)
            res = dfs(nxt, g + 1, bound_val)
            if res == "FOUND":
                return "FOUND"
            if res in ("TIME_LIMIT", "NODE_LIMIT"):
                return res
            if res < minimum:
                minimum = res
            path.pop()
            visited.remove(nxt)
        return minimum

    while True:
        res = dfs(start, 0, bound)
        if res == "FOUND":
            return list(path), expanded, "ok"
        if res == "TIME_LIMIT":
            return None, expanded, "time_limit"
        if res == "NODE_LIMIT":
            return None, expanded, "node_limit"
        if res == INF:
            return None, expanded, "no_solution"
        bound = res


def bfs_limited(start: tuple[int, ...], *, time_limit_sec: float, node_limit: int):
    t0 = time.perf_counter()
    q = [start]
    head = 0
    parent = {start: (None, None)}
    expanded = 0
    peak_open = 1

    while head < len(q):
        if expanded >= node_limit:
            return None, expanded, peak_open, "node_limit"
        if (time.perf_counter() - t0) >= time_limit_sec:
            return None, expanded, peak_open, "time_limit"

        state = q[head]
        head += 1
        expanded += 1
        if state == GOAL_STATE:
            moves = []
            cur = state
            while True:
                prev, mv = parent[cur]
                if prev is None:
                    break
                moves.append(mv)
                cur = prev
            return list(reversed(moves)), expanded, peak_open, "ok"

        for mv, nxt in puzzle_neighbors(state):
            if nxt not in parent:
                parent[nxt] = (state, mv)
                q.append(nxt)
        peak_open = max(peak_open, len(q) - head)

    return None, expanded, peak_open, "no_solution"


def run_weighted_manhattan(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    weights = [0.5, 1.0, 1.5, 2.0, 3.0]
    rows = []

    for trial in range(args.trials):
        start = random_scramble(args.scramble_depth, rng)

        optimal_path, _, optimal_status = ida_star(
            start,
            heuristic=lambda s: manhattan(s),
            time_limit_sec=args.time_limit,
            node_limit=args.node_limit,
        )
        optimal_len = len(optimal_path) if optimal_status == "ok" and optimal_path is not None else None

        for w in weights:
            t0 = time.perf_counter()
            path, expanded, status = ida_star(
                start,
                heuristic=lambda s, ww=w: ww * manhattan(s),
                time_limit_sec=args.time_limit,
                node_limit=args.node_limit,
            )
            elapsed = time.perf_counter() - t0
            length = len(path) if path is not None else None
            optimality = None
            if length is not None and optimal_len is not None:
                if optimal_len == 0:
                    optimality = 1.0 if length == 0 else None
                else:
                    optimality = length / optimal_len
            rows.append(
                {
                    "study": "weighted_manhattan",
                    "trial": trial,
                    "scramble_depth": args.scramble_depth,
                    "w": w,
                    "elapsed": elapsed,
                    "expanded": expanded,
                    "solution_length": length,
                    "optimal_length": optimal_len,
                    "optimality": optimality,
                    "status": status,
                }
            )

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_linear_conflict(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    betas = [1, 2, 3, 4]
    rows = []

    for trial in range(args.trials):
        start = random_scramble(args.scramble_depth, rng)
        for beta in betas:
            t0 = time.perf_counter()
            path, expanded, status = ida_star(
                start,
                heuristic=lambda s, b=beta: manhattan(s) + b * linear_conflicts(s),
                time_limit_sec=args.time_limit,
                node_limit=args.node_limit,
            )
            elapsed = time.perf_counter() - t0
            rows.append(
                {
                    "study": "linear_conflict",
                    "trial": trial,
                    "scramble_depth": args.scramble_depth,
                    "beta": beta,
                    "elapsed": elapsed,
                    "expanded": expanded,
                    "solution_length": len(path) if path is not None else None,
                    "status": status,
                }
            )

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_scramble_depth(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    depths = [5, 10, 20, 50]
    rows = []

    for depth in depths:
        for trial in range(args.trials):
            start = random_scramble(depth, rng)

            t0 = time.perf_counter()
            ida_path, ida_expanded, ida_status = ida_star(
                start,
                heuristic=manhattan,
                time_limit_sec=args.time_limit,
                node_limit=args.node_limit,
            )
            ida_elapsed = time.perf_counter() - t0

            t0 = time.perf_counter()
            bfs_path, bfs_expanded, bfs_peak_open, bfs_status = bfs_limited(
                start,
                time_limit_sec=args.time_limit,
                node_limit=args.node_limit,
            )
            bfs_elapsed = time.perf_counter() - t0

            rows.append(
                {
                    "study": "scramble_depth",
                    "depth": depth,
                    "trial": trial,
                    "algorithm": "IDA*",
                    "elapsed": ida_elapsed,
                    "expanded": ida_expanded,
                    "solution_length": len(ida_path) if ida_path else None,
                    "peak_open_size": None,
                    "status": ida_status,
                }
            )
            rows.append(
                {
                    "study": "scramble_depth",
                    "depth": depth,
                    "trial": trial,
                    "algorithm": "BFS",
                    "elapsed": bfs_elapsed,
                    "expanded": bfs_expanded,
                    "solution_length": len(bfs_path) if bfs_path else None,
                    "peak_open_size": bfs_peak_open,
                    "status": bfs_status,
                }
            )

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="15-puzzle research runner")
    parser.add_argument("study", choices=["weighted_manhattan", "linear_conflict", "scramble_depth"])
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--scramble-depth", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--node-limit", type=int, default=500000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    set_puzzle_size(args.size)

    if args.study == "weighted_manhattan":
        run_weighted_manhattan(args)
    elif args.study == "linear_conflict":
        run_linear_conflict(args)
    else:
        run_scramble_depth(args)


if __name__ == "__main__":
    main()
