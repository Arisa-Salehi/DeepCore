"""Post-correction independent replay/cost benchmark; invoked with PYTHONPATH=src.
Only completed executions are serialized; a timeout is NOT marked PASS.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from deepcore import solver, heuristics as h
from deepcore.environment import DeepCoreEnv

OUT = Path(__file__).resolve().parent / 'final_benchmark.json'
records = []

for i in range(1, 6):
    env = DeepCoreEnv.from_file_or_builtin(f'level{i}')
    start = env.get_initial_state()
    hvals = [h.h_naive_manhattan(start,env),h._mst_payload_bound(start,env,wall_aware=False),h.h1_basic(start,env),h._mst_payload_bound(start,env,wall_aware=True),h.h2_advanced(start,env)]
    methods = [('BFS',lambda: solver.breadth_first_search(env)),
               ('UCS',lambda: solver.uniform_cost_search(env)),
               ('A0',lambda: solver.a_star_search(env,h.h_naive_manhattan)),
               ('A1',lambda: solver.a_star_search(env,h.h1_basic)),
               ('A2',lambda: solver.a_star_search(env,h.h2_advanced)),
               ('GBFS',lambda: solver.greedy_best_first_search(env,h.h2_advanced)),
               ('IDA',lambda: solver.ida_star_search(env,h.h2_advanced)),
               ('WA1.5',lambda: solver.weighted_a_star_search(env,h.h2_advanced,1.5)),
               ('WA2',lambda: solver.weighted_a_star_search(env,h.h2_advanced,2.0)),
               ('WA3',lambda: solver.weighted_a_star_search(env,h.h2_advanced,3.0)),
               ('SMA2000',lambda: solver.sma_star_search(env,h.h2_advanced,2000))]
    print(f'LEVEL {i} H0,MST,H1,WALL_MST,H2={hvals}',flush=True)
    for name,fn in methods:
        t=time.monotonic()
        result=fn()
        runtime=round(time.monotonic()-t,4)
        replay_cost=0
        cur=start
        try:
            for action in result.solution_actions:
                cur, step=env.transition(cur,action)
                replay_cost+=step
            replay_ok=bool(result.solved and env.is_goal(cur) and replay_cost==result.total_cost)
        except Exception:
            replay_ok=False
        d={'level':f'level{i}','algorithm':name,'solved':result.solved,
           'steps':len(result.solution_actions),'cost':result.total_cost,
           'expanded':result.nodes_expanded,'generated':result.nodes_generated,
           'frontier':result.max_frontier_size,'memory':result.max_memory_size,
           'status':result.status_message,'runtime_sec':runtime,'replay_ok':replay_ok,
           'initial_h':hvals}
        records.append(d)
        OUT.write_text(json.dumps(records,indent=2))
        print(f'  {name:<8} solved={result.solved} cost={result.total_cost} replay={replay_ok} expanded={result.nodes_expanded} frontier={result.max_frontier_size} time={runtime:.2f}s',flush=True)
        if not replay_ok:
            raise AssertionError(d)
