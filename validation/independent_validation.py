"""Independent black-box validations. Run from project root after `pip install -e . --no-deps`:

    python -m unittest discover -s validation -p 'independent_validation.py' -v

Original teacher tests and infrastructure files are never modified by this script.
"""
from __future__ import annotations
import heapq
import math
import random
import unittest
from collections import deque
from itertools import count

from deepcore.environment import DeepCoreEnv
from deepcore import solver, heuristics


def replay(env, result):
    assert result is not None and result.solved, (result and result.status_message)
    s = env.get_initial_state()
    cost = 0
    for a in result.solution_actions:
        assert a in [x.value for x in env.get_legal_actions(s)], (a, s)
        s, delta = env.transition(s, a)
        cost += delta
    assert env.is_goal(s), s
    assert result.total_cost == cost, (result.total_cost, cost)
    return len(result.solution_actions), cost


def reachable(env, budget=3000):
    q = deque([env.get_initial_state()]); seen={q[0]}; edges=[]
    while q and len(seen) < budget:
        s=q.popleft()
        for action, t, c in env.get_successors(s):
            edges.append((s,t,c))
            if t not in seen:
                seen.add(t);q.append(t)
    return list(seen), edges


class FromStateEnv:
    def __init__(self, env, start):
        self.env, self.start = env, start
    def get_initial_state(self):
        return self.start
    def __getattr__(self, attr):
        return getattr(self.env, attr)


def small_random_envs(n=10):
    rng = random.Random(20261114)
    for j in range(n):
        grid = [['#']*7 for _ in range(7)]
        positions = [(r,c) for r in range(1,6) for c in range(1,6)]
        for r,c in positions:
            grid[r][c] = '#' if rng.random()<.12 else '.'
        free=[p for p in positions if grid[p[0]][p[1]]=='.']
        rng.shuffle(free)
        count_miners=1 + (j%2)
        elements=['S','E']+['M']*count_miners+['B']*(j%2)+['O']*(j%2)+['@']*(j%3==0)+['~']*(1+(j%2))
        for marker,(r,c) in zip(elements,free):grid[r][c]=marker
        yield DeepCoreEnv('\n'.join(''.join(row) for row in grid), level_name=f'RANDOM_{j}')


class IndependentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maps = [DeepCoreEnv.from_file_or_builtin('level'+str(i)) for i in range(1,6)]

    def test_builtin_all_algorithms_replay_and_optimality(self):
        for env in self.maps:
            with self.subTest(level=env.level_name):
                bfs=solver.breadth_first_search(env)
                ucs=solver.uniform_cost_search(env)
                shortest,cost = replay(env,bfs)
                _,optimal = replay(env,ucs)
                for fn in (solver.depth_first_search, solver.iterative_deepening_search,
                           solver.bidirectional_search):
                    got=fn(env)
                    n,c=replay(env,got)
                    if fn != solver.depth_first_search:
                        self.assertEqual(n,shortest,(env.level_name,fn.__name__))
                for h in (heuristics.h_naive_manhattan, heuristics.h1_basic,
                          heuristics.h2_advanced):
                    m=solver.a_star_search(env,h)
                    self.assertEqual(replay(env,m)[1], optimal)
                for algo in (solver.greedy_best_first_search,solver.ida_star_search):
                    m=algo(env,heuristics.h2_advanced)
                    _,c=replay(env,m)
                    if algo==solver.ida_star_search:
                        self.assertEqual(c,optimal)
                for w in (1.5,2.0,3.0):
                    m=solver.weighted_a_star_search(env,heuristics.h2_advanced,weight=w)
                    _,c=replay(env,m)
                    self.assertLessEqual(c,w*optimal+1e-9)
                m=solver.sma_star_search(env,heuristics.h2_advanced,memory_limit=2000)
                self.assertEqual(replay(env,m)[1],optimal)
                self.assertLessEqual(m.max_memory_size,2000)

    def test_consistency_dominance_all_sampled_edges(self):
        for env in self.maps:
            states,edges=reachable(env,3000)
            for s in states:
                h0=heuristics.h_naive_manhattan(s,env)
                h1=heuristics.h1_basic(s,env)
                h2=heuristics.h2_advanced(s,env)
                self.assertGreaterEqual(h0,0)
                self.assertLessEqual(h0,h1)
                self.assertLessEqual(h1,h2)
                if env.is_goal(s): self.assertEqual((h1,h2),(0,0))
            for s,t,c in edges:
                for h in (heuristics.h1_basic,heuristics.h2_advanced):
                    self.assertLessEqual(h(s,env), c+h(t,env),(env.level_name,s,t,c,h.__name__))

    def test_admissibility_from_intermediate_states(self):
        for env in self.maps:
            s=env.get_initial_state()
            # Pick positions along a valid path, including altered obstacles.
            path=solver.a_star_search(env,heuristics.h2_advanced).solution_actions
            candidates=[s]
            for index,act in enumerate(path):
                s,_=env.transition(s,act)
                if index in (len(path)//3, 2*len(path)//3,len(path)-1): candidates.append(s)
            for sample in candidates:
                h1=heuristics.h1_basic(sample,env)
                h2=heuristics.h2_advanced(sample,env)
                true=solver.uniform_cost_search(FromStateEnv(env,sample))
                self.assertTrue(true.solved)
                self.assertLessEqual(h1,true.total_cost,(env.level_name,sample))
                self.assertLessEqual(h2,true.total_cost,(env.level_name,sample))

    def test_random_custom_maps_exact_reference_and_unsolvable(self):
        for env in small_random_envs():
            with self.subTest(map=env.level_name):
                bfs=solver.breadth_first_search(env)
                ucs=solver.uniform_cost_search(env)
                a1=solver.a_star_search(env,heuristics.h1_basic)
                a2=solver.a_star_search(env,heuristics.h2_advanced)
                self.assertEqual(bfs.solved,ucs.solved)
                self.assertEqual(a1.solved,ucs.solved)
                self.assertEqual(a2.solved,ucs.solved)
                if ucs.solved:
                    self.assertEqual(replay(env,a1)[1],ucs.total_cost)
                    self.assertEqual(replay(env,a2)[1],ucs.total_cost)
                    ids=solver.iterative_deepening_search(env)
                    bi=solver.bidirectional_search(env)
                    self.assertEqual(replay(env,ids)[0],len(bfs.solution_actions))
                    self.assertEqual(replay(env,bi)[0],len(bfs.solution_actions))
                    ida=solver.ida_star_search(env, heuristics.h2_advanced, max_expansions=100000)
                    self.assertEqual(replay(env,ida)[1],ucs.total_cost)
                    sma=solver.sma_star_search(env, heuristics.h2_advanced, memory_limit=2000)
                    self.assertEqual(replay(env,sma)[1],ucs.total_cost)
                    self.assertLessEqual(sma.max_memory_size,2000)
                states,edges=reachable(env,1000)
                for s,t,c in edges:
                    for h in (heuristics.h1_basic,heuristics.h2_advanced):
                        self.assertLessEqual(h(s,env),c+h(t,env))

    def test_expansion_and_depth_limits(self):
        env=self.maps[0]
        for fn in (solver.depth_first_search,solver.iterative_deepening_search,
                   solver.ida_star_search):
            args={'max_expansions':0}
            if fn==solver.ida_star_search:
                result=fn(env,heuristics.h2_advanced,**args)
            else:result=fn(env,**args)
            self.assertFalse(result.solved)
            self.assertIn('limit',result.status_message.lower())
        result=solver.iterative_deepening_search(env,max_depth=5)
        self.assertFalse(result.solved)

    def test_unsolvable_and_cycle_handling(self):
        env=DeepCoreEnv("#########\n#S..#M.E#\n#...#...#\n#########")
        algorithms=[solver.breadth_first_search,solver.depth_first_search,
                    solver.iterative_deepening_search,solver.bidirectional_search,
                    solver.uniform_cost_search]
        for fn in algorithms:
            r=fn(env)
            self.assertFalse(r.solved,fn.__name__)
        for fn in (solver.greedy_best_first_search,solver.a_star_search,
                   solver.ida_star_search,solver.sma_star_search):
            r=fn(env,heuristics.h2_advanced)
            self.assertFalse(r.solved,fn.__name__)

    def test_sma_impossible_memory_path_bound(self):
        for env in self.maps:
            r=solver.sma_star_search(env,heuristics.h2_advanced,memory_limit=10)
            self.assertFalse(r.solved)
            self.assertIn('memory',r.status_message.lower())
            self.assertLessEqual(r.max_memory_size,10)

    def test_h2_drilling_pushing_bridge_and_corner_cases(self):
        maps=[
            '########\n#S.BO.E#\n#...M..#\n########',
            '########\n#S@..E.#\n#.#M~..#\n########',
            '########\n#SB.O.E#\n#.@M~..#\n########',
            '########\n#S#.#E##\n###M####\n########',
            '########\n#S..E..#\n#M.M@~.#\n########',
        ]
        for text in maps:
            env=DeepCoreEnv(text)
            states,edges=reachable(env,2000)
            for s,t,c in edges:
                for h in (heuristics.h1_basic,heuristics.h2_advanced):
                    self.assertLessEqual(h(s,env),c+h(t,env))
            u=solver.uniform_cost_search(env)
            a=solver.a_star_search(env,heuristics.h2_advanced)
            self.assertEqual(u.solved,a.solved)
            if u.solved:self.assertEqual(replay(env,a)[1],u.total_cost)

if __name__=='__main__':unittest.main(verbosity=2)
