"""Reproducible final separate SMA memory sweep and uninformed path audit.
Run: PYTHONPATH=src python validation/perfect_benchmark.py
Original instructor source/tests are not changed.
"""
from __future__ import annotations
import json
from pathlib import Path
from deepcore.environment import DeepCoreEnv
from deepcore import solver,heuristics

folder=Path(__file__).resolve().parent
rows=[]
for i in range(1,6):
    env=DeepCoreEnv.from_file_or_builtin('level'+str(i))
    for cap in (10,30,48,64,128,512,2000):
        result=solver.sma_star_search(env,heuristics.h2_advanced,cap)
        s=env.get_initial_state();cost=0;valid=False
        try:
            for act in result.solution_actions:
                s,delta=env.transition(s,act);cost+=delta
            valid=result.solved and env.is_goal(s) and cost==result.total_cost
        except Exception:
            valid=False
        assert result.max_memory_size<=cap
        if result.solved:assert valid
        row={'level':i,'memory_limit':cap,'solved':result.solved,
             'steps':len(result.solution_actions),'cost':result.total_cost,
             'expanded':result.nodes_expanded,'generated':result.nodes_generated,
             'frontier':result.max_frontier_size,'memory_peak':result.max_memory_size,
             'status':result.status_message,'replay_valid':bool(valid)}
        rows.append(row)
        print('SMA',i,cap,row['solved'],row['cost'],row['expanded'],row['memory_peak'],row['status'],flush=True)
        (folder/'perfect_sma_sweep.json').write_text(json.dumps(rows,indent=2))

path_audit=[]
for i in range(1,6):
    env=DeepCoreEnv.from_file_or_builtin('level'+str(i))
    bfs=solver.breadth_first_search(env)
    assert bfs.solved
    for label,fn in [('DFS',solver.depth_first_search),('IDS',solver.iterative_deepening_search),
                     ('Bidirectional',solver.bidirectional_search)]:
        r=fn(env)
        s=env.get_initial_state();cost=0
        for act in r.solution_actions:
            s,c=env.transition(s,act);cost+=c
        assert r.solved and env.is_goal(s) and r.total_cost==cost
        if label!='DFS': assert len(r.solution_actions)==len(bfs.solution_actions)
        item={'level':i,'algorithm':label,'solved':True,
              'steps':len(r.solution_actions),'cost':r.total_cost,
              'expanded':r.nodes_expanded,'generated':r.nodes_generated,
              'frontier':r.max_frontier_size,'memory':r.max_memory_size,'replay_valid':True}
        path_audit.append(item)
        (folder/'perfect_uninformed_benchmark.json').write_text(json.dumps(path_audit,indent=2))
        print('PATH',i,label,item['steps'],item['cost'],item['expanded'],flush=True)
