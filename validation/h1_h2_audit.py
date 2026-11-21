"""Reproducible H1/H2 verification against actual environment transitions.

From project root, with PYTHONPATH=src:
  python validation/h1_h2_audit.py properties
  python validation/h1_h2_audit.py benchmark

Property exploration is breadth-first and LIMITED to 6000 discovered states
per built-in map; this is NOT exhaustive proof of concrete-state reachability.
The mathematical consistency proof is in Report.pdf.
"""
from __future__ import annotations
import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from deepcore import heuristics as H, solver
from deepcore.environment import DeepCoreEnv

HERE = Path(__file__).resolve().parent


def reachable_edges(env, limit=6000):
    initial = env.get_initial_state()
    seen = {initial}
    queue = deque([initial])
    while queue and len(seen) < limit:
        s = queue.popleft()
        for _, nxt, cost in env.get_successors(s):
            yield s, nxt, cost
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    # Instrumentation of discovered/explored counts is handled separately.


def property_audit(limit=6000):
    records = []
    for i in range(1, 6):
        env = DeepCoreEnv.from_file_or_builtin('level'+str(i))
        q = deque([env.get_initial_state()]); seen = {q[0]}
        expanded = 0; edges = 0; checked = set(); t0 = time.monotonic()
        def check(s):
            if s in checked:
                return
            checked.add(s)
            h0=H.h_naive_manhattan(s,env)
            m=H._mst_payload_bound(s,env,wall_aware=False)
            w=H._mst_payload_bound(s,env,wall_aware=True)
            h1=H.h1_basic(s,env);h2=H.h2_advanced(s,env)
            assert 0<=h0<=m<=h1<=h2, (env.level_name,s,h0,m,h1,h2)
            assert m<=w<=h2, (env.level_name,s,m,w,h2)
            # R1 and R2 dominance of formula also checked against their
            # actual table values, not just max() wrappers.
            data=H._abstract_data(env)
            parts=H._key_parts(s,data)
            assert parts is not None
            assert m<=data['h1'].get(parts[:2],math.inf)
            assert w<=data['h2'].get(parts,math.inf)
            if env.is_goal(s): assert h1==h2==0
        while q and len(seen)<limit:
            s=q.popleft();expanded+=1
            check(s)
            for _,nxt,c in env.get_successors(s):
                edges+=1;check(nxt)
                assert H.h1_basic(s,env)<=c+H.h1_basic(nxt,env), ('H1',env.level_name,s,nxt,c)
                assert H.h2_advanced(s,env)<=c+H.h2_advanced(nxt,env), ('H2',env.level_name,s,nxt,c)
                if nxt not in seen:
                    seen.add(nxt);q.append(nxt)
        row={'map':env.level_name,'unique_states_discovered':len(seen),
             'unique_states_property_checked':len(checked),'states_expanded':expanded,
             'edges_consistency_checked':edges,'passed':True,
             'runtime_seconds':round(time.monotonic()-t0,3),
             'coverage':'bounded BFS; not exhaustive entire map state-space'}
        records.append(row)
        print('PROPERTY',row,flush=True)
        (HERE/'h1_h2_property_audit.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('TOTAL',sum(r['unique_states_property_checked'] for r in records),
          'state checks',sum(r['edges_consistency_checked'] for r in records),'edge checks')


def replay(env,result):
    if not result.solved: raise AssertionError(('unsolved',env.level_name,result.status_message))
    s=env.get_initial_state();cost=0
    for act in result.solution_actions:
        assert act in [a.value for a in env.get_legal_actions(s)]
        s,d=env.transition(s,act);cost+=d
    assert env.is_goal(s) and cost==result.total_cost
    return cost


def benchmark():
    rec=[]
    for i in range(1,6):
        env=DeepCoreEnv.from_file_or_builtin('level'+str(i));s=env.get_initial_state()
        initials={
          'h0':H.h_naive_manhattan(s,env),'literal_mst':H._mst_payload_bound(s,env,wall_aware=False),
          'h1':H.h1_basic(s,env),'wall_mst':H._mst_payload_bound(s,env,wall_aware=True),
          'h2':H.h2_advanced(s,env)
        }
        d={'map':env.level_name,'initial':initials,'algorithms':{}}
        methods={'UCS':lambda:solver.uniform_cost_search(env),
                 'A*(H0)':lambda:solver.a_star_search(env,H.h_naive_manhattan),
                 'A*(H1)':lambda:solver.a_star_search(env,H.h1_basic),
                 'A*(H2)':lambda:solver.a_star_search(env,H.h2_advanced)}
        for name,fn in methods.items():
            t=time.monotonic();r=fn();dt=time.monotonic()-t
            cost=replay(env,r)
            d['algorithms'][name]={'total_cost':cost,'nodes_expanded':r.nodes_expanded,
               'nodes_generated':r.nodes_generated,'max_frontier_size':r.max_frontier_size,
               'runtime_sec':round(dt,4),'replay_valid':True}
            print('BENCH',env.level_name,name,d['algorithms'][name],flush=True)
        assert len({x['total_cost'] for x in d['algorithms'].values()})==1
        assert initials['h0']<=initials['literal_mst']<=initials['h1']<=initials['h2']<= d['algorithms']['UCS']['total_cost']
        assert d['algorithms']['A*(H2)']['nodes_expanded']<=d['algorithms']['A*(H1)']['nodes_expanded']
        rec.append(d)
        (HERE/'h1_h2_benchmark.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
    print('BENCHMARK complete, all 5 levels have A*(H0/H1/H2) == UCS')


if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ('properties','benchmark'):
        raise SystemExit('Usage: python validation/h1_h2_audit.py properties|benchmark')
    {'properties':property_audit,'benchmark':benchmark}[sys.argv[1]]()
