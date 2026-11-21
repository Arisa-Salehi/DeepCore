"""Consistent DeepCore heuristics derived from explicit *relaxed search problems*.

H0 is the unchanged teacher-provided Manhattan baseline.
H1 solves optimally the wall-aware *empty mine* relaxation with real payload
surcharges. H2 solves a strictly more constrained relaxation: sludge retains
its charge, rubble must be drilled, and a chasm requires a bridge action that
consumes one anonymous boulder. Boulder positions and push feasibility are
relaxed, so every real action is simulated by an abstract edge of no greater
cost. Reverse multi-source Dijkstra computes exact abstract costs-to-go.

The abstraction mapping is a graph simulation and yields H(s) <= c(s,a,s') +
H(s'), proving consistency; goal distance is zero, proving admissibility.
H1 <= H2 because every H2 edge is simulated in H1 at no greater cost.
The assignment's *literal* MST-plus-payload bound is admissible but NOT
consistent (MST can change by more than a single movement cost). We compute
it explicitly, but the exact relaxed H1 always dominates that bound, so the
returned H1 is still consistent. Similarly H2's exact relaxation dominates
its wall-MST bound. See Report.pdf for the counterexample and proofs.
"""
from __future__ import annotations

import heapq
import itertools
import math
import weakref
from collections import deque
from typing import Dict, Tuple

from .environment import DeepCoreEnv, DIRECTION_DELTAS
from .state import DeepCoreState, Pos

INF = math.inf
_ENV_CACHES = weakref.WeakKeyDictionary()
_GEOMETRY_CACHES = weakref.WeakKeyDictionary()


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def h_naive_manhattan(state: DeepCoreState, env: DeepCoreEnv) -> float:
    """UNCHANGED teacher baseline H0: Manhattan to nearest waiting miner or E."""
    if env.is_goal(state):
        return 0.0
    if state.miners:
        return float(min(_manhattan(state.bot_pos, p) for p in state.miners))
    return float(_manhattan(state.bot_pos, env.elevator_pos))


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


def _metric_data(env: DeepCoreEnv):
    """Cache BFS shortest-path lengths through STATIC walls only, per map.

    Dynamic boulders, rubble and chasms are optimistically walkable. Thus
    every real bot MOVE/PUSH changes d_wall(bot,E) by at most one, including
    where a bridge has just been constructed. Results are never cross-map.
    """
    result = _GEOMETRY_CACHES.get(env)
    if result is not None:
        return result
    traversable = {(r, c) for r in range(env.rows) for c in range(env.cols)
                   if (r, c) not in env.walls}
    result = {'cells': traversable, 'distances': {}, 'components': {},
              'mst_bounds': {}}
    _GEOMETRY_CACHES[env] = result
    return result


def _wall_distance(a: Pos, b: Pos, env: DeepCoreEnv) -> float:
    data = _metric_data(env)
    cache = data['distances']
    if a not in cache:
        distances = {a: 0}
        queue = deque([a])
        cells = data['cells']
        while queue:
            p = queue.popleft()
            for dr, dc in DIRECTION_DELTAS.values():
                q = (p[0] + dr, p[1] + dc)
                if q in cells and q not in distances:
                    distances[q] = distances[p] + 1
                    queue.append(q)
        cache[a] = distances
    return float(cache[a].get(b, INF))


def _mst_weight(terminals, distance):
    """Prim's MST in the wall/Manhattan metric; repeated positions collapse."""
    points = tuple(dict.fromkeys(terminals))
    if len(points) < 2:
        return 0.0
    visited = {points[0]}
    total = 0.0
    while len(visited) < len(points):
        edge = min((distance(a, b), b) for a in visited
                   for b in points if b not in visited)
        if not math.isfinite(edge[0]):
            return INF
        total += edge[0]
        visited.add(edge[1])
    return total


def _mst_payload_bound(state: DeepCoreState, env: DeepCoreEnv,
                       *, wall_aware: bool) -> float:
    """The literal assignment MST + independent, unavoidable payload bound.

    MST lower-bounds all future bot locomotion; the 2k bot-E term bounds the
    journey of miners ALREADY carried; each remaining miner contributes its
    own pickup-to-E carrying distance. Each pickup-to-E charge is independent,
    not double-counted: one move with k miners pays the 2 surcharge k times.
    """
    # Only robot position, waiting miners and carried count affect the
    # mathematical expression. Boulder/rubble/bridge changes must not cause
    # repeated evaluation of the same MST and payload terms. The cache is
    # scoped to this environment, preventing cross-map distance contamination.
    cache = _metric_data(env)['mst_bounds']
    key = (wall_aware, state.bot_pos, state.miners, state.carried)
    if key in cache:
        return cache[key]
    dist = ((lambda a, b: _wall_distance(a, b, env)) if wall_aware
            else _manhattan)
    E = env.elevator_pos
    terminals = [state.bot_pos, E, *state.miners]
    tree = _mst_weight(terminals, dist)
    result = float(tree + (2 * state.carried * dist(state.bot_pos, E)
                           if state.carried else 0)
                   + sum(2 * dist(p, E) for p in state.miners))
    cache[key] = result
    return result


def _static_bridge_deadlock(state: DeepCoreState, env: DeepCoreEnv) -> bool:
    """Prove an *entirely* immovable boulder supply cannot bridge a necessary pit.

    A boulder is certified permanently immovable only when EVERY direction
    has either a permanent wall/edge at the robot's pushing position or at
    its destination. Dynamic obstacles are intentionally ignored. A bridge
    is certified necessary only when some required terminal is unreachable
    even with ALL non-wall/non-pit cells treated as open. No heuristic guess
    about which exact boulder must be pushed is used to return infinity.
    """
    if not state.chasms or not state.boulders:
        return False  # lack of boulders is handled by the abstract PDB
    cache = _metric_data(env)
    components = cache['components'].get(state.chasms)
    if components is None:
        clear = cache['cells'] - set(state.chasms)
        components = {}
        next_id = 0
        for seed in clear:
            if seed in components:
                continue
            next_id += 1
            components[seed] = next_id
            queue = deque([seed])
            while queue:
                pos = queue.popleft()
                for dr, dc in DIRECTION_DELTAS.values():
                    nxt = (pos[0] + dr, pos[1] + dc)
                    if nxt in clear and nxt not in components:
                        components[nxt] = next_id
                        queue.append(nxt)
        cache['components'][state.chasms] = components
    home = components.get(state.bot_pos)
    if home is not None and all(components.get(p) == home
                                for p in (*state.miners, env.elevator_pos)):
        return False
    nonwall = _metric_data(env)['cells']
    for r, c in state.boulders:
        for dr, dc in DIRECTION_DELTAS.values():
            if ((r - dr, c - dc) in nonwall and
                    (r + dr, c + dc) in nonwall):
                return False  # this boulder MIGHT be movable in the future
    return True


def h1_basic(state: DeepCoreState, env: DeepCoreEnv) -> float:
    """Admissible consistent strengthened H1: max(specification MST, PDB H1).

    The literal instructor MST-plus-payload expression is evaluated exactly
    via _mst_payload_bound(..., wall_aware=False) and memoized by its relevant
    state fields. We keep the already proven stronger empty-mine pattern
    database for the returned value: the literal expression is admissible
    but inconsistent on a reachable Level 1 edge. The exact relaxation
    pointwise dominates it on reachable states and is consistent, so max()
    returns that relaxation without weakening existing A* performance.
    See Report.pdf for the proof and the strict numerical-rubric tradeoff.
    """
    if env.is_goal(state):
        return 0.0
    data = _abstract_data(env)
    parts = _key_parts(state, data)
    if parts is None:
        return INF
    literal = _mst_payload_bound(state, env, wall_aware=False)
    return float(max(literal, data['h1'].get(parts[:2], INF)))


def h2_advanced(state: DeepCoreState, env: DeepCoreEnv) -> float:
    """H2 combines a wall-aware BFS-MST, barrier PDB, and *certified* deadlock.

    The PDB models DRILL and bridge-building PUSH as actual abstract actions
    rather than adding a speculative surcharge to traversal. This avoids
    counting the same movement step twice. The wall-aware MST itself need NOT
    be consistent; however the exact relaxed PDB dominates it everywhere,
    and so the returned H2 remains consistent. Certified impossibility is
    successor-closed, hence assigning infinity is consistent on deadlocks.
    """
    if env.is_goal(state):
        return 0.0
    data = _abstract_data(env)
    parts = _key_parts(state, data)
    if parts is None:
        return INF
    if _static_bridge_deadlock(state, env):
        return INF
    wall_bound = _mst_payload_bound(state, env, wall_aware=True)
    return float(max(wall_bound, h1_basic(state, env),
                     data['h2'].get(parts, INF)))
