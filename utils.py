import time
import tracemalloc
from dataclasses import dataclass


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
