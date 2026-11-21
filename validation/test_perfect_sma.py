"""Strict post-redesign tests. No teacher code or tests are modified.
Run: PYTHONPATH=src python -m unittest discover -s validation -p test_perfect_sma.py -v
"""
import math
import unittest
from collections import deque
from deepcore.environment import DeepCoreEnv
from deepcore import solver, heuristics
from independent_validation import small_random_envs, reachable, FromStateEnv, replay


class MemoryBoundedBestFirstChecks(unittest.TestCase):
    @staticmethod
    def map(index):
        return DeepCoreEnv.from_file_or_builtin(f'level{index}')

    def test_level2_failed_memory_caps_now_reach_optimum(self):
        env=self.map(2)
        for cap in (48, 64):
            with self.subTest(cap=cap):
                r=solver.sma_star_search(env,heuristics.h2_advanced,cap)
                self.assertTrue(r.solved, r.status_message)
                self.assertEqual(replay(env,r)[1],162)
                self.assertLessEqual(r.max_memory_size, cap)
                self.assertIn('Evictions=',r.status_message)

    def test_five_levels_multiple_memory_budgets(self):
        for i in range(1,6):
            env=self.map(i)
            optimal=solver.a_star_search(env,heuristics.h2_advanced)
            optimal_cost=replay(env,optimal)[1]
            for cap in (48,64,128,512,2000):
                with self.subTest(map=i, cap=cap):
                    r=solver.sma_star_search(env,heuristics.h2_advanced,cap)
                    self.assertLessEqual(r.max_memory_size,cap)
                    if r.solved:
                        self.assertGreaterEqual(replay(env,r)[1],optimal_cost)
                        if cap>=len(optimal.solution_actions)+1:
                            self.assertEqual(r.total_cost,optimal_cost)
                    else:
                        # A resource cutoff must not be mistaken for no solution.
                        self.assertTrue(r.status_message)

    def test_regeneration_and_real_evictions(self):
        for i in (2,3,4,5):
            env=self.map(i)
            r=solver.sma_star_search(env,heuristics.h2_advanced,64)
            self.assertTrue(r.solved, (i,r.status_message))
            self.assertLessEqual(r.max_memory_size,64)
            self.assertRegex(r.status_message,r'Evictions=[1-9]')
            self.assertRegex(r.status_message,r'regenerations=[1-9]')

    def test_shortest_versus_cheapest_paths_memory_tradeoff(self):
        env=self.map(5)
        short=solver.sma_star_search(env,heuristics.h2_advanced,48)
        long=solver.sma_star_search(env,heuristics.h2_advanced,64)
        self.assertEqual(replay(env,short)[1],178)
        self.assertEqual(replay(env,long)[1],174)
        self.assertLessEqual(len(short.solution_actions),47)
        self.assertLessEqual(len(long.solution_actions),63)

    def test_level5_memory48_depth_bounded_optimality_certificate(self):
        """Exact layer DP: best cost among all paths with <=47 actions is 178.

        This formally certifies that 48 resident nodes cannot store any
        cost-174 solution, rather than extrapolating from one A* route.
        """
        env=self.map(5)
        frontier={env.get_initial_state():0}
        best=math.inf
        for depth in range(1,48):
            successor={}
            for s,g in frontier.items():
                for _,n,c in env.get_successors(s):
                    ng=g+c
                    if ng<successor.get(n,math.inf):
                        successor[n]=ng
            frontier=successor
            for state,cost in frontier.items():
                if env.is_goal(state):
                    best=min(best,cost)
        self.assertEqual(best,178)

    def test_small_random_maps_ucs_and_sma(self):
        for env in small_random_envs(12):
            with self.subTest(map=env.level_name):
                ref=solver.uniform_cost_search(env)
                result=solver.sma_star_search(env,heuristics.h2_advanced,500)
                self.assertLessEqual(result.max_memory_size,500)
                self.assertEqual(result.solved,ref.solved)
                if ref.solved:
                    self.assertEqual(replay(env,result)[1],replay(env,ref)[1])

    def test_random_real_edges_and_exact_remaining_costs(self):
        for env in small_random_envs(12):
            states, edges=reachable(env,1000)
            for s,t,c in edges:
                h0=heuristics.h_naive_manhattan(s,env)
                h1=heuristics.h1_basic(s,env)
                h2=heuristics.h2_advanced(s,env)
                self.assertTrue(0<=h0<=h1<=h2)
                self.assertLessEqual(h1,c+heuristics.h1_basic(t,env))
                self.assertLessEqual(h2,c+heuristics.h2_advanced(t,env))
            for s in states[:6]:
                ref=solver.uniform_cost_search(FromStateEnv(env,s))
                actual = ref.total_cost if ref.solved else math.inf
                self.assertLessEqual(heuristics.h1_basic(s,env),actual)
                self.assertLessEqual(heuristics.h2_advanced(s,env),actual)

    def test_tiny_memory_and_one_step_solution(self):
        env=DeepCoreEnv('#####\n#SE.#\n#####')
        r=solver.sma_star_search(env,heuristics.h2_advanced,memory_limit=2)
        self.assertTrue(r.solved,r.status_message)
        self.assertEqual(replay(env,r)[1],1)
        self.assertLessEqual(r.max_memory_size,2)
        long=self.map(2)
        r=solver.sma_star_search(long,heuristics.h2_advanced,memory_limit=10)
        self.assertFalse(r.solved)
        self.assertIn('memory limit',r.status_message)

    def test_weighted_and_ida_costs(self):
        for i in range(1,6):
            env=self.map(i)
            best=solver.a_star_search(env,heuristics.h2_advanced)
            baseline=replay(env,best)[1]
            ida=solver.ida_star_search(env,heuristics.h2_advanced)
            self.assertEqual(replay(env,ida)[1],baseline)
            for w in (1.5,2,3):
                wa=solver.weighted_a_star_search(env,heuristics.h2_advanced,w)
                c=replay(env,wa)[1]
                self.assertGreaterEqual(c,baseline)
                self.assertLessEqual(c,w*baseline)


if __name__=='__main__':unittest.main(verbosity=2)
