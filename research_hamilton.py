import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Iterable


Position = tuple[int, int]


@dataclass
class HamiltonMetrics:
    elapsed: float
    recursive_calls: int
    backtracks: int
    dead_ends: int
    solutions: int
    pruned_states: int
    jumps: int
    depth_reduction: int
    max_depth: int
    status: str


def grid_neighbors(pos: Position, rows: int, cols: int) -> Iterable[Position]:
    r, c = pos
    if r > 0:
        yield r - 1, c
    if r + 1 < rows:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < cols:
        yield r, c + 1


def connectivity_ok(free_cells: set[Position], visited: set[Position], rows: int, cols: int) -> bool:
    unvisited = free_cells - visited
    if not unvisited:
        return True
    start = next(iter(unvisited))
    queue = [start]
    seen = {start}
    i = 0
    while i < len(queue):
        pos = queue[i]
        i += 1
        for nxt in grid_neighbors(pos, rows, cols):
            if nxt in unvisited and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == len(unvisited)


def unvisited_degree(pos: Position, free_cells: set[Position], visited: set[Position], rows: int, cols: int) -> int:
    return sum(1 for nxt in grid_neighbors(pos, rows, cols) if nxt in free_cells and nxt not in visited)


def has_dangerous_state(
    free_cells: set[Position],
    visited: set[Position],
    finish: Position,
    rows: int,
    cols: int,
    path_len: int,
    free_count: int,
) -> bool:
    if path_len >= free_count - 1:
        return False
    unvisited = free_cells - visited
    for cell in unvisited:
        if cell == finish:
            continue
        deg = sum(1 for nxt in grid_neighbors(cell, rows, cols) if nxt in unvisited)
        if deg <= 1:
            return True
    return False


def can_finish(path_len: int, free_count: int) -> bool:
    return path_len == free_count - 1


def parse_pos(value: str) -> Position:
    a, b = value.split(",")
    return int(a), int(b)


def solve_hamilton_dfs(
    rows: int,
    cols: int,
    start: Position,
    finish: Position,
    *,
    warnsdorff_alpha: float = 0.0,
    connectivity_policy: str = "never",
    rng: random.Random,
    time_limit_sec: float,
    node_limit: int,
) -> HamiltonMetrics:
    free_cells = {(r, c) for r in range(rows) for c in range(cols)}
    free_count = len(free_cells)
    visited = {start}
    path = [start]

    recursive_calls = 0
    backtracks = 0
    dead_ends = 0
    solutions = 0
    pruned_states = 0
    max_depth = 1

    start_t = time.perf_counter()
    status = "ok"

    def should_stop() -> bool:
        nonlocal status
        if recursive_calls >= node_limit:
            status = "node_limit"
            return True
        if (time.perf_counter() - start_t) >= time_limit_sec:
            status = "time_limit"
            return True
        return False

    def should_check_connectivity(depth: int) -> bool:
        if connectivity_policy == "never":
            return False
        if connectivity_policy == "1":
            return True
        if connectivity_policy == "2":
            return depth % 2 == 0
        if connectivity_policy == "5":
            return depth % 5 == 0
        if connectivity_policy == "danger":
            return has_dangerous_state(free_cells, visited, finish, rows, cols, len(path), free_count)
        return False

    def ordered_moves(pos: Position) -> list[Position]:
        moves = [n for n in grid_neighbors(pos, rows, cols) if n in free_cells and n not in visited]
        if warnsdorff_alpha > 0:
            moves.sort(
                key=lambda n: unvisited_degree(n, free_cells, visited, rows, cols)
                + warnsdorff_alpha * rng.random()
            )
        else:
            moves.sort(key=lambda n: unvisited_degree(n, free_cells, visited, rows, cols))
        return moves

    def dfs(pos: Position) -> bool:
        nonlocal recursive_calls, backtracks, dead_ends, solutions, pruned_states, max_depth
        recursive_calls += 1
        max_depth = max(max_depth, len(path))
        if should_stop():
            return False
        if len(path) == free_count:
            if pos == finish:
                solutions += 1
            return False
        if should_check_connectivity(len(path)) and not connectivity_ok(free_cells, visited, rows, cols):
            pruned_states += 1
            return False

        moves = ordered_moves(pos)
        if not moves:
            dead_ends += 1
            return False

        for nxt in moves:
            if nxt == finish and not can_finish(len(path), free_count):
                continue
            visited.add(nxt)
            path.append(nxt)
            dfs(nxt)
            path.pop()
            visited.remove(nxt)
            backtracks += 1
            if status != "ok":
                return False
        return False

    dfs(start)
    return HamiltonMetrics(
        elapsed=time.perf_counter() - start_t,
        recursive_calls=recursive_calls,
        backtracks=backtracks,
        dead_ends=dead_ends,
        solutions=solutions,
        pruned_states=pruned_states,
        jumps=0,
        depth_reduction=0,
        max_depth=max_depth,
        status=status,
    )


def solve_hamilton_backjump(
    rows: int,
    cols: int,
    start: Position,
    finish: Position,
    *,
    mode: str,
    rng: random.Random,
    warnsdorff_alpha: float,
    time_limit_sec: float,
    node_limit: int,
) -> HamiltonMetrics:
    free_cells = {(r, c) for r in range(rows) for c in range(cols)}
    free_count = len(free_cells)

    recursive_calls = 0
    dead_ends = 0
    solutions = 0
    jumps = 0
    depth_reduction = 0
    max_depth = 1
    status = "ok"

    start_t = time.perf_counter()

    def ordered_moves(pos: Position, visited: set[Position]) -> list[Position]:
        moves = [n for n in grid_neighbors(pos, rows, cols) if n in free_cells and n not in visited]
        if warnsdorff_alpha > 0:
            moves.sort(
                key=lambda n: unvisited_degree(n, free_cells, visited, rows, cols)
                + warnsdorff_alpha * rng.random()
            )
        else:
            moves.sort(key=lambda n: unvisited_degree(n, free_cells, visited, rows, cols))
        return moves

    visited = {start}
    path = [start]
    stack = [{"pos": start, "moves": ordered_moves(start, visited)}]

    while stack:
        if recursive_calls >= node_limit:
            status = "node_limit"
            break
        if (time.perf_counter() - start_t) >= time_limit_sec:
            status = "time_limit"
            break

        frame = stack[-1]
        pos = frame["pos"]
        recursive_calls += 1
        max_depth = max(max_depth, len(path))

        if len(path) == free_count:
            if pos == finish:
                solutions += 1
            frame["moves"] = []

        if frame["moves"]:
            nxt = frame["moves"].pop(0)
            if nxt == finish and not can_finish(len(path), free_count):
                continue
            visited.add(nxt)
            path.append(nxt)
            stack.append({"pos": nxt, "moves": ordered_moves(nxt, visited)})
            continue

        if len(path) < free_count:
            dead_ends += 1

        current_depth = len(stack) - 1
        if current_depth == 0:
            visited.remove(stack.pop()["pos"])
            path.pop()
            break

        conflict_levels = [
            i
            for i in range(len(stack) - 1)
            if stack[i]["moves"]
        ]

        if mode == "one_level":
            target_depth = current_depth - 1
        elif mode == "last_conflict":
            target_depth = max(conflict_levels) if conflict_levels else current_depth - 1
        else:  # max_conflict
            target_depth = min(conflict_levels) if conflict_levels else current_depth - 1

        target_depth = max(0, min(target_depth, current_depth - 1))
        jump = current_depth - target_depth
        jumps += 1
        depth_reduction += max(0, jump - 1)

        while len(stack) - 1 >= target_depth + 1:
            popped = stack.pop()
            visited.remove(popped["pos"])
            path.pop()

    return HamiltonMetrics(
        elapsed=time.perf_counter() - start_t,
        recursive_calls=recursive_calls,
        backtracks=0,
        dead_ends=dead_ends,
        solutions=solutions,
        pruned_states=0,
        jumps=jumps,
        depth_reduction=depth_reduction,
        max_depth=max_depth,
        status=status,
    )


def write_rows(output: str, fieldnames: list[str], rows: list[dict]) -> None:
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_warnsdorff(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    rows_out = []
    for alpha in [0.0, 0.1, 0.5, 1.0, 2.0]:
        for trial in range(args.trials):
            metrics = solve_hamilton_dfs(
                args.rows,
                args.cols,
                parse_pos(args.start),
                parse_pos(args.finish),
                warnsdorff_alpha=alpha,
                connectivity_policy="never",
                rng=rng,
                time_limit_sec=args.time_limit,
                node_limit=args.node_limit,
            )
            rows_out.append({
                "study": "warnsdorff",
                "alpha": alpha,
                "trial": trial,
                **metrics.__dict__,
            })
    write_rows(args.output, list(rows_out[0].keys()), rows_out)


def run_connectivity(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    rows_out = []
    for policy in ["1", "2", "5", "danger"]:
        for trial in range(args.trials):
            metrics = solve_hamilton_dfs(
                args.rows,
                args.cols,
                parse_pos(args.start),
                parse_pos(args.finish),
                warnsdorff_alpha=0.0,
                connectivity_policy=policy,
                rng=rng,
                time_limit_sec=args.time_limit,
                node_limit=args.node_limit,
            )
            rows_out.append({
                "study": "connectivity",
                "policy": policy,
                "trial": trial,
                **metrics.__dict__,
            })
    write_rows(args.output, list(rows_out[0].keys()), rows_out)


def run_backjumping(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    rows_out = []
    for mode in ["one_level", "last_conflict", "max_conflict"]:
        for trial in range(args.trials):
            metrics = solve_hamilton_backjump(
                args.rows,
                args.cols,
                parse_pos(args.start),
                parse_pos(args.finish),
                mode=mode,
                warnsdorff_alpha=0.0,
                rng=rng,
                time_limit_sec=args.time_limit,
                node_limit=args.node_limit,
            )
            rows_out.append({
                "study": "backjumping",
                "mode": mode,
                "trial": trial,
                **metrics.__dict__,
            })
    write_rows(args.output, list(rows_out[0].keys()), rows_out)


def run_grid_size(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    rows_out = []
    for size in [4, 5, 6, 7]:
        start = (0, 0)
        finish = (size - 1, size - 1)
        for trial in range(args.trials):
            metrics = solve_hamilton_dfs(
                size,
                size,
                start,
                finish,
                warnsdorff_alpha=0.0,
                connectivity_policy="never",
                rng=rng,
                time_limit_sec=args.time_limit,
                node_limit=args.node_limit,
            )
            rows_out.append({
                "study": "grid_size",
                "size": f"{size}x{size}",
                "trial": trial,
                **metrics.__dict__,
            })
    write_rows(args.output, list(rows_out[0].keys()), rows_out)


def run_start_finish(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    configs = [
        ("corner_to_opposite_corner", (0, 0), (3, 3)),
        ("corner_to_center", (0, 0), (1, 1)),
        ("center_to_center", (1, 1), (2, 2)),
        ("neighbor_to_neighbor", (1, 1), (1, 2)),
    ]
    rows_out = []
    for name, start, finish in configs:
        for trial in range(args.trials):
            metrics = solve_hamilton_dfs(
                4,
                4,
                start,
                finish,
                warnsdorff_alpha=0.0,
                connectivity_policy="never",
                rng=rng,
                time_limit_sec=args.time_limit,
                node_limit=args.node_limit,
            )
            rows_out.append({
                "study": "start_finish",
                "config": name,
                "start": start,
                "finish": finish,
                "trial": trial,
                **metrics.__dict__,
            })
    write_rows(args.output, list(rows_out[0].keys()), rows_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hamiltonian research runner")
    parser.add_argument("study", choices=["warnsdorff", "connectivity", "backjumping", "grid_size", "start_finish"])
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--start", type=str, default="0,0")
    parser.add_argument("--finish", type=str, default="3,3")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--node-limit", type=int, default=500000)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    dispatch: dict[str, Callable[[argparse.Namespace], None]] = {
        "warnsdorff": run_warnsdorff,
        "connectivity": run_connectivity,
        "backjumping": run_backjumping,
        "grid_size": run_grid_size,
        "start_finish": run_start_finish,
    }
    dispatch[args.study](args)


if __name__ == "__main__":
    main()
