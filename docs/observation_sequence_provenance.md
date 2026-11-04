# Code Provenance — Continuous Hospital Observation

## 1. Artifact and scope

New project files `hospital_observation_sequence.py` and
`observation_sequence_core.py`, written for the user's ROS2 Humble/J100 Gazebo
project. No source map, room segmentation, Coverage implementation, or Nav2
configuration is replaced. Filenames intentionally omit DN/version labels.

## 2. Directly reused project code

- `hospital_coverage_core.load_bundle`, `Geometry`, `planar_pose`.
- `hospital_zone_visualizer.verify_map`.
- `observation_planning_core`: Settings, SensorModel, ThresholdTracker,
  build_plan/visibility/grid paths, conservative footprint helpers, coverage checks.
  Includes the separately tested start-clearance patch.
- `observation_goal_planner.geometry_matches`, stamp conversion and atomic JSON
  helper. Its read-only input matching pattern was adapted into the new node; the
  original single-room node file is not modified or launched as a child process.

These are prior project implementations generated during this collaboration, not
new copies of an external IPA/ETH planning engine.

## 3. External API references

Official Humble interfaces and Python action-client implementation were inspected:

- https://raw.githubusercontent.com/ros-navigation/navigation2/humble/nav2_msgs/action/ComputePathToPose.action
- https://raw.githubusercontent.com/ros-navigation/navigation2/humble/nav2_msgs/action/NavigateToPose.action
- https://raw.githubusercontent.com/ros2/rclpy/humble/rclpy/rclpy/action/client.py
- https://raw.githubusercontent.com/ros2/rclpy/humble/rclpy/rclpy/__init__.py
- https://raw.githubusercontent.com/ros2/rcl_interfaces/humble/action_msgs/msg/GoalStatus.msg
- https://raw.githubusercontent.com/ros2/rcl_interfaces/humble/action_msgs/srv/CancelGoal.srv

Code uses the documented message fields and asynchronous calls; no entire external
source module was copied. Humble NavigateToPose/ComputePathToPose result layouts
are used, not newer-distro error_code fields. Exact nonzero UUID + zero timestamp
is used for the fallback cancel request; never cancel-all.

## 4. Algorithm provenance

Visibility-based sensor placement and NBV concepts remain those documented in the
existing observation planner. This wrapper adds bounded mission policy, approximate
nearby-room ordering, measured-coverage decisions and Nav2 execution. It does not
install or implement an exact Fraunhofer convexSPP solver, set-cover solver, ETH
3D RRT/NBV engine, LLM planner, or a global coverage optimality guarantee.

## 5. Changes and rationale

- Replace human CLI coordinate forwarding with preserved (x,y,yaw) action goals.
- Reuse same live 27-zone Coverage, including previously sufficient zones.
- Add per-run pose history, viewpoint limits and low-gain deferral.
- Lock one goal; verify path first; require result + odometry stop + new measured
  coverage before next request.
- Dry run by default; three-goal bound is not three-room completion.
- Default finish-current behavior; optional separately testable clearance-gated
  early cancellation at measured 90%.
- On abort, external cancel, session/reset/time/data fault: halt further requests.
- Explicit mocked tests for former rclpy reserved-property/logger mistakes.

## 6. Verification and remaining uncertainty

39 pure tests and 17 fake-client ROS adapter scenarios passed. The original
33 patched planner tests were re-run. Existing approved map/zone fingerprints and
499983-cell denominator loaded successfully. Synthetic offline examples are clearly
labeled. No physical or live ROS2 cancellation, stop distance, timing performance,
mission coverage, false-deferral rate, or real multi-room completion was measured.
The first user test must be a dry run, followed by a bounded supervised Gazebo run.
