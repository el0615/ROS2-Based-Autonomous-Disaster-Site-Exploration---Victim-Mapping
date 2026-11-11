# DN011 v2.4 Code Provenance

This section records the original v2.4 release. File names and change lists
below describe that release; see the consolidation record for the current layout.

- Base: `DN011_ZoneBased_EntryArrivalRecovery_v2.3_20261102.zip`, provided existing full source. The original v2.3 is copied and unchanged.
- User evidence: 2026-11-02 27-zone supervised run, 14/14 succeeded, then `START_CLEARANCE_BLOCKED` in `ZonePlanner.distances()` because 0.35m conservative bridge returned None.
- Modified code: `src/zone_planner.py` (start distance fallback), `src/zone_node.py` (diagnostics/policy identity), `tests/test_policy_planner.py` (synthetic regression tests), add-only installer target, docs, package manifest. No changes to `zb_reference_planning.py`, `zb_reference_sequence.py`, `zone_policy.py`, RPP, Nav2, Coverage, approved maps and Zone masks.
- Reason: the display-only conservative bridge is too restrictive to serve as a mission-fatal precondition; Navfn controls real path planning, not DN-011's circular clearance distance heuristic.
- New fallback: no graph snapping, no fabricated strict safe connectivity. Only samples strict-safe internal target endpoints and uses Euclidean distance for candidate ranking. Actual travel requires a new successful Navfn `ComputePathToPose` validation plus unchanged pre-send local-start footprint and freshness checks.
- Never bypass: global start known-free requirement, goal centerline checking, goal footprint, Zone-mask containment, stop evidence, cancellation and collision detection.
- Prohibited claims: no live Gazebo validation, no unconditional continuation if Local Costmap truly blocked, no proof of collision-free movement or 27-zone completion.

## Consolidation record — 2026-11-11

The shared planning, sequence, and JSON/CSV functions now live in
`scripts/exploration/planning_core.py`, `sequence_core.py`, and
`exploration_io.py`. These replace the duplicated `zb_reference_planning.py`,
`zb_reference_sequence.py`, and `zb_io.py`. Observation-specific adapters retain
their visibility guard and diagnostics behavior. They select their backend per
call; importing an adapter does not replace the shared planner's backend.

The source modules were grouped by responsibility:

| Former module | Current module |
| --- | --- |
| `four_results.py` | `exploration_results.py` |
| `entry_access.py` | `connectivity_audit.py` |
| `connectivity_policy.py` | `zone_policy.py` |
| `blockage_gate.py` | `path_blockage.py` |
| `mission_budget_runtime.py` | `mission_budget.py` |
| Used helpers from `direct_approach.py` | `two_phase.py` |

The unused legacy access-analysis function and unused approach-catalog code
were removed. Report commands are grouped into `navigation/reports/zone_report.py`
and `inspect_results.py`; their existing result schemas remain unchanged.
The versioned checkpoint payloads, saved mission results, map/Nav2 configuration,
and compiled visibility libraries were retained.

Current visibility guard receipts include hashes of the shared dependencies.
Historical audit receipts and their audited observation-sequence source remain
unchanged. Offline verification compared shared calculations, saved reports,
shell arguments and policy environments before and after consolidation. Backend
isolation and dependency-hash rejection have dedicated regression tests. These
checks do not constitute a new live Gazebo run or proof of mission completion.
