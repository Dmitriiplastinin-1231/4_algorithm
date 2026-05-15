from collections import deque

from utils import SolverStats, StopSearch


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


def can_visit_finish(path_length, free_count):
    return path_length == free_count - 1


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
        if len(path) == free_count:
            if pos == finish:
                solutions += 1
            return
        if use_connectivity and not connectivity_ok(free_cells, visited, rows, cols):
            return
        for nxt in ordered_moves(pos):
            if nxt == finish and not can_visit_finish(len(path), free_count):
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
            if nxt == finish and not can_visit_finish(len(path), free_count):
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
