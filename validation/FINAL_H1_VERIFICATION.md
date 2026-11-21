# H1/H2 final scientific and regression evidence (19 November 2026)

## Final decision

The literal instructor equation is computed by `_mst_payload_bound(state, env, wall_aware=False)`, but it is inconsistent on an actual reachable Level 1 MOVE_N edge: 30 > 1 + 28. `h1_basic` consequently retains the previous consistent exact empty-mine relaxation: `max(literal, R1) == R1` for reachable states. It remains admissible and dominates H0. H2 retains its original more constrained relaxation and certified deadlock checks; H1 <= H2. A different, formula-exact H1 would lose global consistency and may degrade A*.

The **only** heuristic code change from the uploaded completed ZIP was to memoize the Manhattan and wall-aware MST-plus-payload expressions by `(wall_aware, bot_pos, miners, carried)` within each environment. No solver, instructor-owned module, instructor test, or baseline H0 function was edited. H1/H2 values were independently identical to the uploaded implementation on 20,138 explored states per heuristic.

## New verified results

- 9/9 instructor tests passed together.
- 15/15 existing `test*.py` supplemental tests passed together.
- 8/8 `independent_validation.py` test methods passed in individually/split executions, including the full five-map multi-algorithm test. Several attempts to run the entire 8-method independent suite as a *single invocation* were interrupted by the execution environment's timeout; no fresh single-invocation pass is claimed. Old report claims of a single-run pass are historical evidence from the uploaded prior package, not this new rerun.
- Newly bounded breadth-first inspection: 24,316 unique discovered/checked concrete states, 55,539 tested legal successor edges, no dominance or consistency violations. Level 1 exhaustively covered all 316 reachable states; Levels 2-5 capped at 6,000 discovered states each. This does not establish exhaustive state-space consistency by enumeration; see the report's mathematical abstraction proof.
- Newly replayed UCS and A* using H0, H1 and H2 on every built-in level. Optimal costs: L1=68, L2=162, L3=154, L4=141, L5=174. Expanded/generated/max-frontier are in `h1_h2_benchmark.json` and all match the uploaded baseline.
- Independent exact UCS cost-to-go checks from selected reachable intermediate states passed (`independent_validation.py`). It is computationally impractical to evaluate `h*(s)` with UCS for all 24,316 states; no such claim is made.

## Reproduce (from the extracted project root)

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m unittest discover -s validation -p 'test*.py' -v
PYTHONPATH=src python -m unittest validation.independent_validation.IndependentChecks.test_builtin_all_algorithms_replay_and_optimality -v
PYTHONPATH=src python -m unittest validation.independent_validation.IndependentChecks.test_admissibility_from_intermediate_states -v
PYTHONPATH=src python -m unittest validation.independent_validation.IndependentChecks.test_consistency_dominance_all_sampled_edges validation.independent_validation.IndependentChecks.test_random_custom_maps_exact_reference_and_unsolvable validation.independent_validation.IndependentChecks.test_expansion_and_depth_limits validation.independent_validation.IndependentChecks.test_unsolvable_and_cycle_handling validation.independent_validation.IndependentChecks.test_sma_impossible_memory_path_bound validation.independent_validation.IndependentChecks.test_h2_drilling_pushing_bridge_and_corner_cases -v
PYTHONPATH=src python validation/h1_h2_audit.py properties
PYTHONPATH=src python validation/h1_h2_audit.py benchmark
```

## Academic grading exposure

The 12-point H1 task retains a *rubric interpretation risk* if literal numeric equality to the inconsistent teacher equation is mechanically required. The mathematical properties and original performance are preserved; score certainty is impossible. The 13-point H2 task is unchanged and all evaluated dominance, obstacle, optimality and bounded memory tests pass, but no guarantee of grader-awarded marks is asserted. H2's abstract-state complexity may be expensive on larger untested custom maps.
