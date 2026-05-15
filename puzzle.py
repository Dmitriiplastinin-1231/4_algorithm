import heapq
from collections import deque

from utils import StopSearch


MIN_PUZZLE_SIZE = 2
MAX_PUZZLE_SIZE = 6
PUZZLE_SIZE = 4
GOAL_STATE = ()
GOAL_POS = {}
BLANK_MOVES = []
BLANK_MOVE_INDEX = []
DISTANCE_TABLE = []


def build_goal_state(size):
    return tuple(list(range(1, size * size)) + [0])


def build_goal_pos(goal_state, size):
    return {value: divmod(idx, size) for idx, value in enumerate(goal_state)}


def build_blank_moves(size):
    total = size * size
    moves = [[] for _ in range(total)]
    move_index = [{} for _ in range(total)]
    for idx in range(total):
        row, col = divmod(idx, size)
        if row > 0:
            moves[idx].append(("U", idx - size))
            move_index[idx]["U"] = idx - size
        if row < size - 1:
            moves[idx].append(("D", idx + size))
            move_index[idx]["D"] = idx + size
        if col > 0:
            moves[idx].append(("L", idx - 1))
            move_index[idx]["L"] = idx - 1
        if col < size - 1:
            moves[idx].append(("R", idx + 1))
            move_index[idx]["R"] = idx + 1
    return moves, move_index


def build_distance_table(goal_pos, size):
    total = size * size
    positions = [divmod(idx, size) for idx in range(total)]
    table = [[0] * total for _ in range(total)]
    for value in range(1, total):
        goal_row, goal_col = goal_pos[value]
        for idx in range(total):
            row, col = positions[idx]
            table[value][idx] = abs(row - goal_row) + abs(col - goal_col)
    return table


def set_puzzle_size(size):
    if size < MIN_PUZZLE_SIZE or size > MAX_PUZZLE_SIZE:
        raise ValueError(
            "Puzzle size must be between "
            f"{MIN_PUZZLE_SIZE} and {MAX_PUZZLE_SIZE}."
        )
    global PUZZLE_SIZE, GOAL_STATE, GOAL_POS, BLANK_MOVES, BLANK_MOVE_INDEX, DISTANCE_TABLE
    PUZZLE_SIZE = size
    GOAL_STATE = build_goal_state(size)
    GOAL_POS = build_goal_pos(GOAL_STATE, size)
    BLANK_MOVES, BLANK_MOVE_INDEX = build_blank_moves(size)
    DISTANCE_TABLE = build_distance_table(GOAL_POS, size)


set_puzzle_size(PUZZLE_SIZE)
INF = float("inf")
SCRAMBLE_MOVES = 120  # Random moves applied to generate a solvable board.
ANIMATION_DELAY_MS = 120
MAX_BACKJUMP_DEPTH = 80


def manhattan(state):
    total = 0
    distances = DISTANCE_TABLE
    for idx, value in enumerate(state):
        if value == 0:
            continue
        total += distances[value][idx]
    return total


def puzzle_neighbors(state):
    idx0 = state.index(0)
    for move, idx in BLANK_MOVES[idx0]:
        new_state = list(state)
        new_state[idx0], new_state[idx] = new_state[idx], new_state[idx0]
        yield move, tuple(new_state)


def apply_move(state, move):
    idx0 = state.index(0)
    idx = BLANK_MOVE_INDEX[idx0].get(move)
    if idx is None:
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

    while depth <= MAX_BACKJUMP_DEPTH:
        if stop_event and stop_event.is_set():
            raise StopSearch()
        path = []
        visited = {start}
        found = dfs(start, depth, visited, path)
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
