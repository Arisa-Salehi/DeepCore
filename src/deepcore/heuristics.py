"""
heuristics.py
Student heuristic starter file for DeepCore:
  - h_naive_manhattan : provided baseline
  - h1_basic          : required student implementation
  - h2_advanced       : required student implementation
"""

import heapq
import itertools
import math
import weakref
from typing import Dict, List, Tuple
from collections import deque
from .environment import DeepCoreEnv, DIRECTION_DELTAS
from .state import DeepCoreState, Pos

# Cache wall-aware shortest path distances per environment instance
_WALL_DIST_CACHE: Dict[int, Dict[Tuple[Pos, Pos], int]] = {}


def _get_wall_aware_distances(env: DeepCoreEnv) -> Dict[Tuple[Pos, Pos], int]:
    """Precomputes all-pairs shortest path steps ignoring dynamic obstacles (only walls # block)."""
    env_id = id(env)
    if env_id in _WALL_DIST_CACHE:
        return _WALL_DIST_CACHE[env_id]

    walkable = [
        (r, c)
        for r in range(env.rows)
        for c in range(env.cols)
        if (r, c) not in env.walls
    ]
    dist_map: Dict[Tuple[Pos, Pos], int] = {}

    for src in walkable:
        q = deque([(src, 0)])
        visited = {src}
        while q:
            curr, d = q.popleft()
            dist_map[(src, curr)] = d
            cr, cc = curr
            for dr, dc in DIRECTION_DELTAS.values():
                nxt = (cr + dr, cc + dc)
                if env.in_bounds(nxt) and nxt not in env.walls and nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, d + 1))

    _WALL_DIST_CACHE[env_id] = dist_map
    return dist_map


def _prim_mst_cost(nodes: List[Pos], dist_fn) -> float:
    """Computes Minimum Spanning Tree weight over `nodes` using `dist_fn(u, v)`."""
    if len(nodes) <= 1:
        return 0.0
    in_mst = {nodes[0]}
    total_weight = 0.0

    while len(in_mst) < len(nodes):
        best_edge = float("inf")
        best_node = None
        for u in in_mst:
            for v in nodes:
                if v not in in_mst:
                    w = dist_fn(u, v)
                    if w < best_edge:
                        best_edge = w
                        best_node = v
        if best_node is None:
            break
        in_mst.add(best_node)
        total_weight += best_edge

    return total_weight


def h_naive_manhattan(state: DeepCoreState, env: DeepCoreEnv) -> float:
    """
    Baseline Naive Heuristic: Manhattan distance to nearest waiting miner (or Elevator if none).
    """
    if env.is_goal(state):
        return 0.0
    r, c = state.bot_pos
    if state.miners:
        return float(min(abs(r - mr) + abs(c - mc) for mr, mc in state.miners))
    er, ec = env.elevator_pos
    return float(abs(r - er) + abs(c - ec))


INF = math.inf
_ENV_CACHES = weakref.WeakKeyDictionary()


def _abstract_data(env: DeepCoreEnv):
    """Compute fixed-size exact reverse shortest paths for both relaxations.

    Keys use (cell-index, waiting-mask) for H1, and additionally
    (rubble-mask, chasm-mask, number-of-free-boulders) for H2. All maps' static
    terrain is fixed. Dijkstra is independent of the dynamic concrete states.
    """
    existing = _ENV_CACHES.get(env)
    if existing is not None:
        return existing

    cells = sorted((r, c) for r in range(env.rows)
                   for c in range(env.cols) if (r, c) not in env.walls)
    index = {p: i for i, p in enumerate(cells)}
    adj = [tuple(index[q] for dr, dc in DIRECTION_DELTAS.values()
                 if (q := (p[0] + dr, p[1] + dc)) in index)
           for p in cells]
    miners = sorted(env.initial_miners)
    rubble = sorted(env.initial_rubble)
    chasms = sorted(env.initial_chasms)
    mbit = {index[p]: 1 << i for i, p in enumerate(miners)}
    rbit = {index[p]: 1 << i for i, p in enumerate(rubble)}
    cbit = {index[p]: 1 << i for i, p in enumerate(chasms)}
    M, R, C, B = len(miners), len(rubble), len(chasms), len(env.initial_boulders)
    elevator = index[env.elevator_pos]
    sludge = {index[p] for p in env.sludge}

    # A reverse MOVE from current position p to previous position q includes
    # two possibilities: the forward MOVE did/did not collect a miner at p.
    def reverse_move_masks(pos, mask):
        bit = mbit.get(pos, 0)
        if bit and (mask & bit):  # a robot cannot stand on an uncollected miner
            return ()
        if bit:
            return (mask, mask | bit)
        return (mask,)

    def reverse_dijkstra(advanced, step_only=False):
        dist = {}
        heap = []
        serial = itertools.count()

        def relax(key, val):
            if val < dist.get(key, INF):
                dist[key] = val
                heapq.heappush(heap, (val, next(serial), key))

        if not advanced:
            relax((elevator, 0), 0)
        else:
            for rm in range(1 << R):
                for cm in range(1 << C):
                    for bc in range(B + 1):
                        relax((elevator, 0, rm, cm, bc), 0)

        while heap:
            d, _, key = heapq.heappop(heap)
            if d != dist.get(key):
                continue
            pos, mask = key[:2]
            if not advanced:
                for prev_mask in reverse_move_masks(pos, mask):
                    payload = M - prev_mask.bit_count()
                    cost = 1 if step_only else 1 + 2 * payload
                    for q in adj[pos]:
                        relax((q, prev_mask), d + cost)
                continue

            _, _, rm, cm, bc = key
            # A forward MOVE enters the current position; entry cells cannot
            # contain uncleared rubble or unbridged chasms.
            pos_clear = not ((rm & rbit.get(pos, 0)) or (cm & cbit.get(pos, 0)))
            if pos_clear:
                for prev_mask in reverse_move_masks(pos, mask):
                    payload = M - prev_mask.bit_count()
                    cost = 1 if step_only else (5 if pos in sludge else 1) + 2 * payload
                    for q in adj[pos]:
                        if not ((rm & rbit.get(q, 0)) or (cm & cbit.get(q, 0))):
                            relax((q, prev_mask, rm, cm, bc), d + cost)

                # Reverse abstract PUSH into a chasm: forward action moves bot
                # from q to pos, opens the cell directly past pos and uses one
                # anonymous available boulder. No positional push constraint.
                if bc < B and not (mbit.get(pos, 0) & mask):
                    for dr, dc in DIRECTION_DELTAS.values():
                        prev_pos = (cells[pos][0] - dr, cells[pos][1] - dc)
                        beyond = (cells[pos][0] + dr, cells[pos][1] + dc)
                        q = index.get(prev_pos)
                        z = index.get(beyond)
                        if q is None or z is None:
                            continue
                        cb = cbit.get(z, 0)
                        if not cb or (cm & cb):
                            continue
                        if (rm & rbit.get(q, 0)) or (cm & cbit.get(q, 0)):
                            continue
                        cost = 1 if step_only else 6 + 2 * (M - mask.bit_count())
                        relax((q, mask, rm, cm | cb, bc + 1), d + cost)

            # Reverse DRILL: robot stays put, undo removal of an adjacent rubble.
            for q in adj[pos]:
                rb = rbit.get(q, 0)
                if rb and not (rm & rb):
                    relax((pos, mask, rm | rb, cm, bc), d + (1 if step_only else 10))

        return dist

    data = {"index": index, "mbits": {p: 1 << i for i, p in enumerate(miners)},
            "rbits": {p: 1 << i for i, p in enumerate(rubble)},
            "cbits": {p: 1 << i for i, p in enumerate(chasms)},
            "h1": reverse_dijkstra(False), "h2": reverse_dijkstra(True),
            "step2": reverse_dijkstra(True, step_only=True)}
    _ENV_CACHES[env] = data
    return data


def _key_parts(state: DeepCoreState, data):
    pos = data["index"].get(state.bot_pos)
    if pos is None:
        return None
    try:
        m = sum(data["mbits"][p] for p in state.miners)
        r = sum(data["rbits"][p] for p in state.rubble)
        c = sum(data["cbits"][p] for p in state.chasms)
    except KeyError:
        return None
    return pos, m, r, c, len(state.boulders)


def h1_basic(state: DeepCoreState, env: DeepCoreEnv) -> float:
    """TODO: Build an admissible MST-plus-payload lower bound; return 0 at a goal."""
    raise NotImplementedError


def _is_corner_deadlock(pos: Pos, env: DeepCoreEnv) -> bool:
    """Returns True if a boulder at `pos` is wedged against perpendicular rock walls (#)."""
    r, c = pos
    n_wall = (r - 1, c) in env.walls
    s_wall = (r + 1, c) in env.walls
    e_wall = (r, c + 1) in env.walls
    w_wall = (r, c - 1) in env.walls
    return (n_wall or s_wall) and (e_wall or w_wall)


def h2_advanced(state: DeepCoreState, env: DeepCoreEnv) -> float:
    """TODO: Strengthen H1 while preserving admissibility and dominance."""
    raise NotImplementedError
