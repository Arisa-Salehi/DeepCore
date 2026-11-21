"""Post-correction tests: original teacher tests are never edited.
Run: python -m unittest discover -s validation -p 'test_final_properties.py' -v
"""
import math
import unittest
from collections import deque
from deepcore.environment import DeepCoreEnv
from deepcore import heuristics as H, solver as S


class FinalPropertyTests(unittest.TestCase):
    def test_literal_mst_bounds_and_returned_heuristics_consistent(self):
        edge_count = 0
        for i in range(1, 6):
            env = DeepCoreEnv.from_file_or_builtin(f'level{i}')
            start = env.get_initial_state()
            seen = {start}
            queue = deque([start])
            while queue and len(seen) < 1250:
                s = queue.popleft()
                man = H._mst_payload_bound(s, env, wall_aware=False)
                wall = H._mst_payload_bound(s, env, wall_aware=True)
                h1, h2 = H.h1_basic(s, env), H.h2_advanced(s, env)
                self.assertLessEqual(H.h_naive_manhattan(s, env), man)
                self.assertLessEqual(man, wall)
                self.assertLessEqual(wall, h2)
                self.assertLessEqual(man, h1)
                self.assertLessEqual(h1, h2)
                for _, t, cost in env.get_successors(s):
                    edge_count += 1
                    self.assertLessEqual(h1, cost + H.h1_basic(t, env))
                    self.assertLessEqual(h2, cost + H.h2_advanced(t, env))
                    if t not in seen:
                        seen.add(t)
                        queue.append(t)
        self.assertGreater(edge_count, 10000)

    def test_literal_mst_inconsistency_counterexample_is_real_transition(self):
        env = DeepCoreEnv.from_file_or_builtin('level1')
        s = env.get_initial_state()
        # Reach the concrete state at (2,4), with neither miner collected.
        from collections import deque
        queue = deque([s])
        seen = {s}
        while queue:
            st = queue.popleft()
            if st.bot_pos == (2, 4) and st.carried == 0 and st.rubble == s.rubble:
                break
            for _, child, _ in env.get_successors(st):
                if child not in seen:
                    seen.add(child)
                    queue.append(child)
        else:
            self.fail('Counterexample state not reachable')
        next_state, step_cost = env.transition(st, 'MOVE_N')
        man_before = H._mst_payload_bound(st, env, wall_aware=False)
        man_after = H._mst_payload_bound(next_state, env, wall_aware=False)
        self.assertEqual((man_before, step_cost, man_after), (30, 1, 28))
        self.assertGreater(man_before, step_cost + man_after)
        self.assertLessEqual(H.h1_basic(st, env), step_cost + H.h1_basic(next_state, env))

    def test_permanent_wall_corner_boulder_and_mandatory_chasm_deadlock(self):
        text = '#########\n#SB#...M#\n#..#...E#\n#..O....#\n#########'
        env = DeepCoreEnv(text)
        self.assertTrue(H._static_bridge_deadlock(env.get_initial_state(), env))
        self.assertTrue(math.isinf(H.h2_advanced(env.get_initial_state(), env)))
        self.assertFalse(S.uniform_cost_search(env).solved)

    def test_apparent_corner_deadlock_has_alternate_passage(self):
        text = '#########\n#SB#...M#\n#..#...E#\n#.......#\n#...O...#\n#########'
        env = DeepCoreEnv(text)
        self.assertFalse(H._static_bridge_deadlock(env.get_initial_state(), env))
        self.assertTrue(math.isfinite(H.h2_advanced(env.get_initial_state(), env)))
        ucs = S.uniform_cost_search(env)
        self.assertTrue(ucs.solved)
        self.assertEqual(ucs.total_cost, S.a_star_search(env, H.h2_advanced).total_cost)

    def test_zero_carried_disconnected_map_has_infinity_not_nan(self):
        env = DeepCoreEnv('#########\n#S..#M.E#\n#...#...#\n#########')
        start = env.get_initial_state()
        self.assertEqual(H._mst_payload_bound(start, env, wall_aware=True), math.inf)
        self.assertEqual(H.h2_advanced(start, env), math.inf)

    def test_sma_small_memory_real_eviction_and_valid_replay(self):
        env = DeepCoreEnv.from_file_or_builtin('level1')
        result = S.sma_star_search(env, H.h2_advanced, memory_limit=30)
        self.assertTrue(result.solved)
        self.assertLessEqual(result.max_memory_size, 30)
        self.assertIn('Evictions=', result.status_message)
        state = env.get_initial_state()
        total = 0
        for act in result.solution_actions:
            state, cost = env.transition(state, act)
            total += cost
        self.assertTrue(env.is_goal(state))
        self.assertEqual(total, 68)
        self.assertEqual(result.total_cost, 68)

if __name__ == '__main__':
    unittest.main(verbosity=2)
