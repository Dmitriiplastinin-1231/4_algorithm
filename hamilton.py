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


def build_neighbors(rows, cols):
    neighbors = {}
    for r in range(rows):
        for c in range(cols):
            pos = (r, c)
            neighbors[pos] = list(grid_neighbors(pos, rows, cols))
    return neighbors


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


def unvisited_degree(pos, unvisited, neighbors):
    return sum(1 for neighbor in neighbors[pos] if neighbor in unvisited)


def parity_ok(unvisited_counts, current_pos, finish, colors):
    current_color = colors[current_pos]
    finish_color = colors[finish]
    total_even = unvisited_counts[0] + (1 if current_color == 0 else 0)
    total_odd = unvisited_counts[1] + (1 if current_color == 1 else 0)
    if current_color == finish_color:
        if current_color == 0:
            return total_even == total_odd + 1
        return total_odd == total_even + 1
    return total_even == total_odd


def degree_pruning_ok(unvisited, current_pos, finish, neighbors):
    if not unvisited:
        return current_pos == finish
    for pos in unvisited:
        degree = 0
        for neighbor in neighbors[pos]:
            if neighbor in unvisited or neighbor == current_pos:
                degree += 1
        if degree == 0:
            return False
        if degree == 1 and pos not in (current_pos, finish):
            return False
    current_degree = sum(1 for neighbor in neighbors[current_pos] if neighbor in unvisited)
    if current_degree == 0 and len(unvisited) > 0:
        return False
    return True


def can_visit_finish(path_length, free_count):
    return path_length == free_count - 1


def solve_hamilton(rows, cols, start, finish, blocked, mode, stop_event=None):
    neighbors = build_neighbors(rows, cols)
    free_cells = {
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (r, c) not in blocked
    }
    if start not in free_cells or finish not in free_cells:
        return SolverStats(0, None, 0, 0, 0.0, 0.0)

    free_count = len(free_cells)
    colors = {cell: (cell[0] + cell[1]) % 2 for cell in free_cells}
    unvisited = set(free_cells)
    unvisited.remove(start)
    unvisited_counts = [0, 0]
    for cell in unvisited:
        unvisited_counts[colors[cell]] += 1
    path = [start]
    solutions = 0
    nodes = 0

    use_warnsdorff = mode == "Warnsdorff"
    use_connectivity = mode == "Connectivity pruning"

    def ordered_moves(pos):
        moves = [n for n in neighbors[pos] if n in unvisited]
        if use_warnsdorff:
            moves.sort(key=lambda n: unvisited_degree(n, unvisited, neighbors))
        return moves

    def visit(cell):
        unvisited.remove(cell)
        unvisited_counts[colors[cell]] -= 1
        path.append(cell)

    def unvisit(cell):
        path.pop()
        unvisited_counts[colors[cell]] += 1
        unvisited.add(cell)

    def prune_ok(pos):
        if not parity_ok(unvisited_counts, pos, finish, colors):
            return False
        if not degree_pruning_ok(unvisited, pos, finish, neighbors):
            return False
        if use_connectivity and not connectivity_ok(unvisited, neighbors):
            return False
        return True

    def backtrack(pos):
        nonlocal solutions, nodes
        if stop_event and stop_event.is_set():
            raise StopSearch()
        nodes += 1
        if len(path) == free_count:
            if pos == finish:
                solutions += 1
            return
        if not prune_ok(pos):
            return
        for nxt in ordered_moves(pos):
            if nxt == finish and not can_visit_finish(len(path), free_count):
                continue
            visit(nxt)
            backtrack(nxt)
            unvisit(nxt)

    def backjumping():
        nonlocal solutions, nodes
        backjumps = 0
        start_moves = ordered_moves(start)
        stack = [
            {
                "pos": start,
                "moves": start_moves,
                "index": 0,
                "moves_len": len(start_moves),
            }
        ]
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
            if not prune_ok(pos):
                stack.pop()
                unvisit(pos)
                continue
            if frame["index"] >= frame["moves_len"]:
                stack.pop()
                unvisit(pos)
                jump_count = 0
                while stack and stack[-1]["index"] >= stack[-1]["moves_len"]:
                    frame = stack.pop()
                    unvisit(frame["pos"])
                    jump_count += 1
                if jump_count:
                    backjumps += jump_count
                continue
            nxt = frame["moves"][frame["index"]]
            frame["index"] += 1
            if nxt == finish and not can_visit_finish(len(path), free_count):
                continue
            visit(nxt)
            nodes += 1
            moves = ordered_moves(nxt)
            stack.append(
                {
                    "pos": nxt,
                    "moves": moves,
                    "index": 0,
                    "moves_len": len(moves),
                }
            )
        return backjumps

    backjumps = 0
    if mode == "Backjumping":
        backjumps = backjumping()
    else:
        backtrack(start)
    return SolverStats(solutions, None, nodes, backjumps, 0.0, 0.0)
