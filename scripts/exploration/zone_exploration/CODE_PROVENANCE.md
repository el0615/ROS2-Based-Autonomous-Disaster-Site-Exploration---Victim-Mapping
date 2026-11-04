# DN011 v2.4 Code Provenance

- Base: `DN011_ZoneBased_EntryArrivalRecovery_v2.3_20261102.zip`, provided existing full source. The original v2.3 is copied and unchanged.
- User evidence: 2026-11-02 27-zone supervised run, 14/14 succeeded, then `START_CLEARANCE_BLOCKED` in `ZonePlanner.distances()` because 0.35m conservative bridge returned None.
- Modified code: `src/zone_planner.py` (start distance fallback), `src/zone_node.py` (diagnostics/policy identity), `tests/test_policy_planner.py` (synthetic regression tests), add-only installer target, docs, package manifest. No changes to `zb_reference_planning.py`, `zb_reference_sequence.py`, `zone_policy.py`, RPP, Nav2, Coverage, approved maps and Zone masks.
- Reason: the display-only conservative bridge is too restrictive to serve as a mission-fatal precondition; Navfn controls real path planning, not DN-011's circular clearance distance heuristic.
- New fallback: no graph snapping, no fabricated strict safe connectivity. Only samples strict-safe internal target endpoints and uses Euclidean distance for candidate ranking. Actual travel requires a new successful Navfn `ComputePathToPose` validation plus unchanged pre-send local-start footprint and freshness checks.
- Never bypass: global start known-free requirement, goal centerline checking, goal footprint, Zone-mask containment, stop evidence, cancellation and collision detection.
- Prohibited claims: no live Gazebo validation, no unconditional continuation if Local Costmap truly blocked, no proof of collision-free movement or 27-zone completion.
