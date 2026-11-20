"""
solver.py
Student implementation file for DeepCore search algorithms.

- Breadth-First Search (BFS) is provided below as a complete reference example
  showing how to interact with `env.get_successors(state)` and `SearchMetrics`.
- Implement the remaining required algorithms. Do not change the public
  function signatures or the environment/state modules.
"""

from collections import deque
import heapq
import itertools
import math
from typing import Callable, Dict, List, Optional, Set, Tuple

from .environment import DeepCoreEnv
from .state import DeepCoreState, SearchMetrics


def _reconstruct_path(
    parent_map: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]],
    goal_state: DeepCoreState,
) -> List[str]:
    """Helper function to trace actions from the initial state to `goal_state`."""
    actions: List[str] = []
    curr: Optional[DeepCoreState] = goal_state
    while curr is not None:
        prev, act = parent_map.get(curr, (None, None))
        if act is not None:
            actions.append(act)
        curr = prev
    actions.reverse()
    return actions


# =============================================================================
# 1. Breadth-First Search (BFS) — [PROVIDED AS REFERENCE EXAMPLE]
# =============================================================================
def breadth_first_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    """
    Reference Graph-Search BFS implementation (FIFO queue).
    Minimizes the number of steps, but ignores terrain/payload energy costs!
    """
    metrics = SearchMetrics(algorithm_name="BFS")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics

    frontier = deque([start])
    frontier_set = {start}
    explored: Set[DeepCoreState] = set()
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
    cost_so_far: Dict[DeepCoreState, int] = {start: 0}

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))
        state = frontier.popleft()
        frontier_set.discard(state)

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            if next_state not in explored and next_state not in frontier_set:
                parent[next_state] = (state, action.value)
                cost_so_far[next_state] = cost_so_far[state] + step_cost
                if env.is_goal(next_state):
                    metrics.solved = True
                    metrics.solution_actions = _reconstruct_path(parent, next_state)
                    metrics.total_cost = cost_so_far[next_state]
                    return metrics
                frontier.append(next_state)
                frontier_set.add(next_state)
                metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

    metrics.status_message = "No solution found."
    return metrics





# -----------------------------------------------------------------------------
# Small shared helpers (student code)
# -----------------------------------------------------------------------------
LIMIT_MESSAGE = "Expansion limit reached."
INF = math.inf


def _finish_solved(
    metrics: SearchMetrics,
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]],
    goal: DeepCoreState,
    cost: int,
) -> SearchMetrics:
    metrics.solved = True
    metrics.solution_actions = _reconstruct_path(parent, goal)
    metrics.total_cost = cost
    metrics.status_message = "Solved."
    return metrics


def _replay_cost(env: DeepCoreEnv, actions: List[str]) -> int:
    """Total cost of `actions` replayed from the initial state (single source of truth)."""
    state = env.get_initial_state()
    total = 0
    for act in actions:
        state, c = env.transition(state, act)
        total += c
    return total


class _SuccessorCache:
    """Memoises ``env.get_successors`` (used by the iterative-deepening searches)."""

    def __init__(self, env: DeepCoreEnv, max_entries: int = 400_000):
        self.env = env
        self.max_entries = max_entries
        self.table: Dict[DeepCoreState, List[Tuple[str, DeepCoreState, int]]] = {}

    def get(self, state: DeepCoreState) -> List[Tuple[str, DeepCoreState, int]]:
        hit = self.table.get(state)
        if hit is not None:
            return hit
        succ = [(a.value, s, c) for a, s, c in self.env.get_successors(state)]
        if len(self.table) < self.max_entries:
            self.table[state] = succ
        return succ


# =============================================================================
# 2. Depth-First Search (DFS) & Iterative Deepening Search (IDS)
# =============================================================================
def depth_first_search(env: DeepCoreEnv, max_expansions: Optional[int] = None) -> Optional[SearchMetrics]:
    """
    Depth-First Graph Search with a LIFO stack and an explored set.

    A state is marked explored when it is popped (true DFS order); the goal test
    is also applied at pop time.  Not optimal and (with a graph-search explored
    set) complete on finite state spaces.
    """
    metrics = SearchMetrics(algorithm_name="DFS")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics

    # stack entries: (state, parent_state, action_str, g)
    stack: List[Tuple[DeepCoreState, Optional[DeepCoreState], Optional[str], int]] = [(start, None, None, 0)]
    explored: Set[DeepCoreState] = set()
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {}

    while stack:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(stack))
        metrics.max_memory_size = max(metrics.max_memory_size, len(stack) + len(explored))
        state, prev, act, g = stack.pop()
        if state in explored:
            continue
        explored.add(state)
        parent[state] = (prev, act)

        if env.is_goal(state):
            return _finish_solved(metrics, parent, state, g)

        if max_expansions is not None and metrics.nodes_expanded >= max_expansions:
            metrics.status_message = LIMIT_MESSAGE
            return metrics

        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)
        # reversed() so that the first successor is explored first
        for action, nxt, step_cost in reversed(env.get_successors(state)):
            metrics.nodes_generated += 1
            if nxt not in explored:
                stack.append((nxt, state, action.value, g + step_cost))

    metrics.status_message = "No solution found."
    return metrics


def iterative_deepening_search(
    env: DeepCoreEnv,
    max_depth: Optional[int] = None,
    max_expansions: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """
    Iterative Deepening Search (depth = number of actions).

    An admissible reverse-relaxed lower bound on the remaining *action count*
    prunes nodes with depth + lower_bound > current limit. This is an exact
    depth-safe optimization, not a weighted-cost heuristic. Each iteration restarts from scratch with a larger depth limit L and runs a
    depth-limited DFS.  Inside one iteration a state is only re-expanded when it
    is reached at a strictly smaller depth than before (a transposition table
    that also subsumes the "states on the current path" cycle check), which
    keeps the search depth-optimal while avoiding exponential blow-up on this
    graph-shaped state space.  The goal test is applied at generation time, so
    the first goal found in iteration L has depth exactly L.
    """
    metrics = SearchMetrics(algorithm_name="IDS")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics

    succ_cache = _SuccessorCache(env)
    # Admissible relaxed *action-count* pattern database for depth pruning.
    # It counts DRILL and PUSH as one action each and is a lower bound for every
    # concrete path. It never changes the shortest-step objective of IDS.
    from .heuristics import _abstract_data, _key_parts
    data = _abstract_data(env)
    step_lb = data["step2"]
    def remaining_steps(s):
        key = _key_parts(s, data)
        return step_lb.get(key, INF) if key is not None else INF

    initial_lb = remaining_steps(start)
    if initial_lb == INF:
        metrics.status_message = "No solution found."
        return metrics
    limit = int(initial_lb)
    while True:
        if max_depth is not None and limit > max_depth:
            metrics.status_message = LIMIT_MESSAGE
            return metrics

        best_depth: Dict[DeepCoreState, int] = {start: 0}
        parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
        cost: Dict[DeepCoreState, int] = {start: 0}
        stack: List[Tuple[DeepCoreState, int]] = [(start, 0)]
        cutoff = False

        while stack:
            metrics.max_frontier_size = max(metrics.max_frontier_size, len(stack))
            metrics.max_memory_size = max(metrics.max_memory_size, len(stack) + len(best_depth))
            state, depth = stack.pop()
            if best_depth[state] < depth:
                continue  # a shallower arrival superseded this entry
            lb = remaining_steps(state)
            if depth + lb > limit:
                if lb < INF:
                    cutoff = True
                continue
            if max_expansions is not None and metrics.nodes_expanded >= max_expansions:
                metrics.status_message = LIMIT_MESSAGE
                return metrics
            metrics.nodes_expanded += 1
            metrics.explored_order.append(state.bot_pos)

            successors = succ_cache.get(state)
            if depth >= limit:
                # A finite cyclic graph can have outgoing edges forever. Only
                # a genuinely NEW (or shallower-reached) state warrants a
                # deeper iteration; otherwise unsolvable IDS terminates.
                if any(nxt not in best_depth or best_depth[nxt] > depth + 1
                       for _act, nxt, _cost in successors):
                    cutoff = True
                continue
            for act, nxt, step_cost in reversed(successors):
                metrics.nodes_generated += 1
                nd = depth + 1
                old = best_depth.get(nxt)
                if old is not None and old <= nd:
                    continue
                best_depth[nxt] = nd
                parent[nxt] = (state, act)
                cost[nxt] = cost[state] + step_cost
                if env.is_goal(nxt):
                    actions = _reconstruct_path(parent, nxt)
                    return _finish_solved(metrics, parent, nxt, _replay_cost(env, actions))
                stack.append((nxt, nd))

        if not cutoff:
            metrics.status_message = "No solution found."
            return metrics
        limit += 1


# =============================================================================
# 3. Bidirectional Search
# =============================================================================
def bidirectional_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    """
    TODO: Implement Bidirectional Search expanding forward from `env.get_initial_state()`
    via `env.get_successors(s)` and backward from `env.get_goal_states()` via `env.get_predecessors(s)`.
    """
    raise NotImplementedError


# =============================================================================
# 4-6. Best-first family: UCS, GBFS, A*, Weighted A*
# =============================================================================
def uniform_cost_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    """
    Uniform Cost Search: priority queue ordered by path cost g(n).

    Lazy deletion handles the decrease-key operation (a cheaper path to a state
    already in the queue simply pushes a second entry).  The goal test is
    applied when a node is popped, which is what guarantees optimality.
    """
    metrics = SearchMetrics(algorithm_name="UCS")
    start = env.get_initial_state()
    counter = itertools.count()
    frontier: List[Tuple[int, int, DeepCoreState]] = [(0, next(counter), start)]
    best_g: Dict[DeepCoreState, int] = {start: 0}
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
    explored: Set[DeepCoreState] = set()

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))
        g, _, state = heapq.heappop(frontier)
        if state in explored or g > best_g[state]:
            continue
        if env.is_goal(state):
            return _finish_solved(metrics, parent, state, g)
        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, nxt, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            ng = g + step_cost
            if nxt in explored:
                continue
            if nxt not in best_g or ng < best_g[nxt]:
                best_g[nxt] = ng
                parent[nxt] = (state, action.value)
                heapq.heappush(frontier, (ng, next(counter), nxt))

    metrics.status_message = "No solution found."
    return metrics




def greedy_best_first_search(
    env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    """TODO: Implement Greedy Best-First Search ordering the priority queue strictly by h(n)."""
    raise NotImplementedError


def a_star_search(
    env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    """TODO: Implement A* Graph Search ordering the priority queue by f(n) = g(n) + h(n)."""
    raise NotImplementedError


# =============================================================================
# BONUS 1: Weighted A* Search
# =============================================================================
def weighted_a_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    weight: float = 1.5,
) -> Optional[SearchMetrics]:
    """TODO (Bonus): Implement Weighted A* ordering the frontier by f(n) = g(n) + W * h(n)."""
    raise NotImplementedError


# =============================================================================
# BONUS 2: Iterative Deepening A* (IDA*)
# =============================================================================
def ida_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    max_expansions: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """TODO (Bonus): Implement Iterative Deepening A* (IDA*) using f-cost threshold contours."""
    raise NotImplementedError


# =============================================================================
# BONUS 3: Simplified Memory-Bounded A* (SMA*)
# =============================================================================




def sma_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    memory_limit: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """TODO (Bonus): Implement Simplified Memory-Bounded A* (SMA*)."""
    raise NotImplementedError


