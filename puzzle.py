import heapq
from collections import deque

from utils import StopSearch


MIN_PUZZLE_SIZE = 2
MAX_PUZZLE_SIZE = 6
PUZZLE_SIZE = 4
GOAL_STATE = ()
GOAL_POS = {}
NEIGHBOR_TABLE = []
MANHATTAN_TABLE = []


def build_goal_state(size):
    return tuple(list(range(1, size * size)) + [0])


def build_goal_pos(goal_state, size):
    return {value: divmod(idx, size) for idx, value in enumerate(goal_state)}


def build_neighbor_table(size):
    table = []
    for idx in range(size * size):
        row, col = divmod(idx, size)
        moves = []
        if row > 0:
            moves.append(("U", idx - size))
        if row < size - 1:
            moves.append(("D", idx + size))
        if col > 0:
            moves.append(("L", idx - 1))
        if col < size - 1:
            moves.append(("R", idx + 1))
        table.append(moves)
    return table


def build_manhattan_table(size, goal_pos):
    table = [[0] * (size * size) for _ in range(size * size)]
    for value in range(1, size * size):
        goal_row, goal_col = goal_pos[value]
        for idx in range(size * size):
            row, col = divmod(idx, size)
            table[value][idx] = abs(row - goal_row) + abs(col - goal_col)
    return table


def set_puzzle_size(size):
    if size < MIN_PUZZLE_SIZE or size > MAX_PUZZLE_SIZE:
        raise ValueError(
            "Puzzle size must be between "
            f"{MIN_PUZZLE_SIZE} and {MAX_PUZZLE_SIZE}."
        )
    global PUZZLE_SIZE, GOAL_STATE, GOAL_POS, NEIGHBOR_TABLE, MANHATTAN_TABLE
    PUZZLE_SIZE = size
    GOAL_STATE = build_goal_state(size)
    GOAL_POS = build_goal_pos(GOAL_STATE, size)
    NEIGHBOR_TABLE = build_neighbor_table(size)
    MANHATTAN_TABLE = build_manhattan_table(size, GOAL_POS)


set_puzzle_size(PUZZLE_SIZE)
INF = float("inf")
SCRAMBLE_MOVES = 120  # Random moves applied to generate a solvable board.
ANIMATION_DELAY_MS = 120
MAX_BACKJUMP_DEPTH = 80


def manhattan(state):
    total = 0
    for idx, value in enumerate(state):
        if value == 0:
            continue
        total += MANHATTAN_TABLE[value][idx]
    return total


def puzzle_neighbors(state):
    idx0 = state.index(0)
    for move, idx in NEIGHBOR_TABLE[idx0]:
        new_state = list(state)
        new_state[idx0], new_state[idx] = new_state[idx], new_state[idx0]
        yield move, tuple(new_state)


def apply_move(state, move):
    idx0 = state.index(0)
    row, col = divmod(idx0, PUZZLE_SIZE)
    if move == "U":
        idx = idx0 - PUZZLE_SIZE
    elif move == "D":
        idx = idx0 + PUZZLE_SIZE
    elif move == "L":
        idx = idx0 - 1
    elif move == "R":
        idx = idx0 + 1
    else:
        raise ValueError(f"Unknown move: {move}")
    new_state = list(state)
    new_state[idx0], new_state[idx] = new_state[idx], new_state[idx0]
    return tuple(new_state)


def is_solvable(state):
    if len(state) != PUZZLE_SIZE * PUZZLE_SIZE:
        raise ValueError(
            "Expected a "
            f"{PUZZLE_SIZE}x{PUZZLE_SIZE} puzzle state "
            f"({PUZZLE_SIZE * PUZZLE_SIZE} tiles)."
        )
    values = [v for v in state if v != 0]
    inversions = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[i] > values[j]:
                inversions += 1
    blank_row_from_bottom = PUZZLE_SIZE - (state.index(0) // PUZZLE_SIZE)
    if PUZZLE_SIZE % 2 == 1:
        return inversions % 2 == 0
    return (blank_row_from_bottom % 2 == 0) != (inversions % 2 == 0)


def solve_puzzle_astar(start, stop_event=None):
    open_heap = []
    g_score = {start: 0}
    came_from = {start: (None, None)}
    h_start = manhattan(start)
    h_score = {start: h_start}
    heapq.heappush(open_heap, (h_start, 0, start))
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
        current_h = h_score[state]
        idx0 = state.index(0)
        for move, idx in NEIGHBOR_TABLE[idx0]:
            tile = state[idx]
            nxt = list(state)
            nxt[idx0], nxt[idx] = nxt[idx], nxt[idx0]
            nxt = tuple(nxt)
            ng = g + 1
            if ng < g_score.get(nxt, INF):
                g_score[nxt] = ng
                came_from[nxt] = (state, move)
                new_h = current_h + (MANHATTAN_TABLE[tile][idx0] - MANHATTAN_TABLE[tile][idx])
                h_score[nxt] = new_h
                heapq.heappush(open_heap, (ng + new_h, ng, nxt))
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
        idx0 = state.index(0)
        for move, idx in NEIGHBOR_TABLE[idx0]:
            nxt = list(state)
            nxt[idx0], nxt[idx] = nxt[idx], nxt[idx0]
            nxt = tuple(nxt)
            if nxt not in came_from:
                came_from[nxt] = (state, move)
                queue.append(nxt)
    return None, nodes


def solve_puzzle_ida(start, stop_event=None):
    bound = manhattan(start)
    path = []
    visited = {start}
    nodes = 0

    def dfs(state, g, bound, h):
        nonlocal nodes
        if stop_event and stop_event.is_set():
            raise StopSearch()
        nodes += 1
        f = g + h
        if f > bound:
            return f
        if state == GOAL_STATE:
            return "FOUND"
        minimum = INF
        idx0 = state.index(0)
        for move, idx in NEIGHBOR_TABLE[idx0]:
            tile = state[idx]
            nxt = list(state)
            nxt[idx0], nxt[idx] = nxt[idx], nxt[idx0]
            nxt = tuple(nxt)
            if nxt in visited:
                continue
            visited.add(nxt)
            path.append(move)
            new_h = h + (MANHATTAN_TABLE[tile][idx0] - MANHATTAN_TABLE[tile][idx])
            result = dfs(nxt, g + 1, bound, new_h)
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
        result = dfs(start, 0, bound, bound)
        if result == "FOUND":
            return list(path), nodes
        if result == INF:
            return None, nodes
        bound = result


def solve_puzzle_backjumping(start, stop_event=None):
    depth = 0
    nodes_total = 0
    backjumps = 0

    def ordered_neighbors(state, h):
        idx0 = state.index(0)
        moves = []
        for move, idx in NEIGHBOR_TABLE[idx0]:
            tile = state[idx]
            nxt = list(state)
            nxt[idx0], nxt[idx] = nxt[idx], nxt[idx0]
            nxt = tuple(nxt)
            new_h = h + (MANHATTAN_TABLE[tile][idx0] - MANHATTAN_TABLE[tile][idx])
            moves.append((new_h, move, nxt))
        moves.sort(key=lambda item: item[0])
        return [(move, nxt, new_h) for new_h, move, nxt in moves]

    def dfs(state, depth_limit, visited, path, h):
        nonlocal nodes_total, backjumps
        if stop_event and stop_event.is_set():
            raise StopSearch()
        nodes_total += 1
        if state == GOAL_STATE:
            return True
        if depth_limit == 0:
            return False
        any_branch = False
        for move, nxt, new_h in ordered_neighbors(state, h):
            if nxt in visited:
                continue
            any_branch = True
            visited.add(nxt)
            path.append(move)
            if dfs(nxt, depth_limit - 1, visited, path, new_h):
                return True
            path.pop()
            visited.remove(nxt)
        if not any_branch and depth_limit > 0:
            backjumps += 1
        return False

    while depth <= MAX_BACKJUMP_DEPTH:
        if stop_event and stop_event.is_set():
            raise StopSearch()
        path = []
        visited = {start}
        found = dfs(start, depth, visited, path, manhattan(start))
        if found:
            return path, nodes_total, backjumps
        depth += 1
    return None, nodes_total, backjumps


def build_path(came_from, state):
    moves = []
    while True:
        parent, move = came_from[state]
        if parent is None:
            break
        moves.append(move)
        state = parent
    return list(reversed(moves))
