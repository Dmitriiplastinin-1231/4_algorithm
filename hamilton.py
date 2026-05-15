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


def connectivity_ok(unvisited, neighbors):
    if not unvisited:
        return True
    start = next(iter(unvisited))
    queue = deque([start])
    seen = {start}
    while queue:
        pos = queue.popleft()
        for neighbor in neighbors[pos]:
            if neighbor in unvisited and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen) == len(unvisited)


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

    neighbors = {
        cell: [
            n
            for n in grid_neighbors(cell, rows, cols)
            if n in free_cells
        ]
        for cell in free_cells
    }

    color_counts = [0, 0]
    for row, col in free_cells:
        color_counts[(row + col) & 1] += 1
    diff = color_counts[0] - color_counts[1]
    start_color = (start[0] + start[1]) & 1
    finish_color = (finish[0] + finish[1]) & 1
    if abs(diff) > 1:
        return SolverStats(0, None, 0, 0, 0.0, 0.0)
    if diff == 0 and start_color == finish_color:
        return SolverStats(0, None, 0, 0, 0.0, 0.0)
    if diff == 1 and not (start_color == finish_color == 0):
        return SolverStats(0, None, 0, 0, 0.0, 0.0)
    if diff == -1 and not (start_color == finish_color == 1):
        return SolverStats(0, None, 0, 0, 0.0, 0.0)

    free_count = len(free_cells)
    visited = {start}
    unvisited = set(free_cells)
    unvisited.remove(start)
    remaining_degree = {cell: len(neighbors[cell]) for cell in free_cells}
    path = [start]
    solutions = 0
    nodes = 0

    use_warnsdorff = mode == "Warnsdorff"
    use_connectivity = mode == "Connectivity pruning"

    def visit(cell):
        visited.add(cell)
        unvisited.remove(cell)
        path.append(cell)
        for nbr in neighbors[cell]:
            if nbr in unvisited:
                remaining_degree[nbr] -= 1

    def unvisit(cell):
        visited.remove(cell)
        path.pop()
        unvisited.add(cell)
        count = 0
        for nbr in neighbors[cell]:
            if nbr in unvisited:
                remaining_degree[nbr] += 1
                count += 1
        remaining_degree[cell] = count

    def degree_pruning(pos):
        forced = None
        remaining = len(unvisited)
        for cell in unvisited:
            deg = remaining_degree[cell]
            if cell == finish:
                if deg == 0 and remaining > 1:
                    return False, None
                continue
            if deg == 0:
                return False, None
            if deg == 1:
                if forced is None:
                    forced = cell
                else:
                    return False, None
        if forced is not None and forced not in neighbors[pos]:
            return False, None
        return True, forced

    def ordered_moves(pos):
        moves = [n for n in neighbors[pos] if n in unvisited]
        if use_warnsdorff:
            moves.sort(key=lambda n: remaining_degree[n])
        if finish in moves and not can_visit_finish(len(path), free_count):
            moves = [n for n in moves if n != finish]
        return moves

    def available_moves(pos):
        ok, forced = degree_pruning(pos)
        if not ok:
            return []
        if use_connectivity and not connectivity_ok(unvisited, neighbors):
            return []
        moves = ordered_moves(pos)
        if forced is not None:
            if forced in moves:
                return [forced]
            return []
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
        for nxt in available_moves(pos):
            visit(nxt)
            backtrack(nxt)
            unvisit(nxt)

    def backjumping():
        nonlocal solutions, nodes
        backjumps = 0
        stack = [{"pos": start, "moves": available_moves(start)}]
        while stack:
            if stop_event and stop_event.is_set():
                raise StopSearch()
            frame = stack[-1]
            pos = frame["pos"]
            if len(path) == free_count:
                if pos == finish:
                    solutions += 1
                stack.pop()
                unvisit(pos)
                continue
            if not frame["moves"]:
                stack.pop()
                unvisit(pos)
                jump_count = 0
                while stack and not stack[-1]["moves"]:
                    frame = stack.pop()
                    unvisit(frame["pos"])
                    jump_count += 1
                if jump_count:
                    backjumps += jump_count
                continue
            nxt = frame["moves"].pop(0)
            visit(nxt)
            nodes += 1
            stack.append({"pos": nxt, "moves": available_moves(nxt)})
        return backjumps

    visit(start)
    for nbr in neighbors[start]:
        if nbr in unvisited:
            remaining_degree[nbr] -= 1
    backjumps = 0
    if mode == "Backjumping":
        backjumps = backjumping()
    else:
        backtrack(start)
    return SolverStats(solutions, None, nodes, backjumps, 0.0, 0.0)
