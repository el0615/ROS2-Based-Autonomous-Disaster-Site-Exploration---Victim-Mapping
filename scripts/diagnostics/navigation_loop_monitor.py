#!/usr/bin/env python3
"""Navigation loop monitor: Gazebo / ROS 2 Humble.

The monitor detects repeated spatial returns together with increasing Nav2
recovery counts. By default it only observes and writes CSV/JSON evidence.
With --auto-cancel, it requests a UUID-specific cancel on the first
LOOP_CANDIDATE. This is a navigation policy, not proof of unreachability or
an emergency-stop mechanism. At most one cancel request is sent per process.

Usage:
  python3 navigation_loop_monitor.py --namespace /j100_0001 --output-dir ./logs/navigation_monitor --auto-cancel
  python3 navigation_loop_monitor.py --self-test
  python3 navigation_loop_monitor.py --replay-csv previous_navigation.csv

Launch the monitor before sending a NEW RViz Goal. Already-running goals are
observe-only. Ctrl+C stops only this monitor; it does not cancel navigation.
The Nav2 controller configuration and velocity commands are never modified.

ROS 2 Humble API contracts:
https://raw.githubusercontent.com/ros2/rcl_interfaces/humble/action_msgs/srv/CancelGoal.srv
https://raw.githubusercontent.com/ros2/rcl_interfaces/humble/action_msgs/msg/GoalStatus.msg
https://raw.githubusercontent.com/ros-navigation/navigation2/humble/nav2_msgs/action/NavigateToPose.action
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSION = "0.2.2-candidate"
STATUS_NAMES = {0: "UNKNOWN", 1: "ACCEPTED", 2: "EXECUTING", 3: "CANCELING",
                4: "SUCCEEDED", 5: "CANCELED", 6: "ABORTED"}
TERMINAL = {4, 5, 6}

@dataclass(frozen=True)
class Settings:
    sample_seconds: float = 1.0
    return_radius_m: float = 0.35
    departure_radius_m: float = 1.0
    minimum_return_travel_m: float = 2.0
    minimum_return_seconds: float = 8.0
    warning_returns: int = 3
    warning_recovery_increase: int = 2
    jump_distance_m: float = 1.0
    jump_speed_mps: float = 3.0
    maximum_feedback_gap_seconds: float = 5.0
    maximum_anchors: int = 5000


@dataclass(frozen=True)
class Sample:
    goal_id: str
    frame_id: str
    navigation_s: float
    pose_stamp_s: float
    x: float
    y: float
    yaw_rad: float
    distance_remaining_m: float
    recoveries: int


@dataclass
class Anchor:
    x: float
    y: float
    first_recoveries: int
    last_visit_s: float
    last_visit_travel_m: float
    returns: int = 0
    departed: bool = False


class GoalHistory:
    """Per-goal approximate spatial-return detector; no ROS dependencies."""

    def __init__(self, config: Settings):
        self.config = config
        self.previous: Sample | None = None
        self.anchors: list[Anchor] = []
        self.travel_m = 0.0
        self.segment = 0

    def _reset(self, sample: Sample) -> None:
        self.segment += 1
        self.previous = sample
        self.travel_m = 0.0
        self.anchors = [Anchor(sample.x, sample.y, sample.recoveries,
                               sample.navigation_s, 0.0)]

    def update(self, sample: Sample) -> dict[str, Any] | None:
        cfg = self.config
        result: dict[str, Any] = {
            "event": "RECORDING", "step_m": 0.0, "travel_m": self.travel_m,
            "anchor_returns": 0, "anchor_recovery_increase": 0,
            "anchor_x": "", "anchor_y": "", "note": "", "segment": self.segment,
        }
        if self.previous is None:
            self._reset(sample)
            result.update(event="START", segment=self.segment,
                          note="History starts here; earlier motion is not reconstructed.")
            return result

        old = self.previous
        elapsed = sample.navigation_s - old.navigation_s
        reset_reason = ""
        if sample.goal_id != old.goal_id:
            reset_reason = "goal_changed"
        elif sample.frame_id != old.frame_id:
            reset_reason = "frame_changed"
        elif elapsed < -1e-6:
            reset_reason = "navigation_time_went_backwards"
        elif sample.recoveries < old.recoveries:
            reset_reason = "recovery_counter_reset"
        elif elapsed < cfg.sample_seconds - 1e-6:
            return None  # Same/stale timestamps and high-rate feedback are not revisits.
        elif sample.pose_stamp_s <= old.pose_stamp_s:
            reset_reason = "pose_stamp_not_advancing"
        elif elapsed > cfg.maximum_feedback_gap_seconds:
            reset_reason = "feedback_gap; intervening motion unknown"

        step = math.hypot(sample.x - old.x, sample.y - old.y)
        if (not reset_reason and elapsed > 0.0 and
                step > cfg.jump_distance_m and step / elapsed > cfg.jump_speed_mps):
            reset_reason = "pose_discontinuity_suspected; not classified as a loop"

        if reset_reason:
            self._reset(sample)
            result.update(event="RESET", travel_m=0.0, segment=self.segment,
                          note=reset_reason)
            return result

        self.travel_m += step
        self.previous = sample
        result.update(step_m=step, travel_m=self.travel_m, segment=self.segment)
        nearest: Anchor | None = None
        nearest_distance = math.inf
        for anchor in self.anchors:
            distance = math.hypot(sample.x - anchor.x, sample.y - anchor.y)
            if distance >= cfg.departure_radius_m:
                anchor.departed = True
            if distance < nearest_distance:
                nearest, nearest_distance = anchor, distance

        if nearest is None or nearest_distance > cfg.return_radius_m:
            if len(self.anchors) < cfg.maximum_anchors:
                self.anchors.append(Anchor(sample.x, sample.y, sample.recoveries,
                                           sample.navigation_s, self.travel_m))
            else:
                result.update(event="ANCHOR_LIMIT", note="New anchors not added; raw logging continues.")
            return result

        anchor = nearest
        result.update(anchor_x=anchor.x, anchor_y=anchor.y,
                      anchor_returns=anchor.returns,
                      anchor_recovery_increase=max(0, sample.recoveries - anchor.first_recoveries))
        if not anchor.departed:
            # Hovering/rotating at the same spot is not a spatial return.
            anchor.last_visit_s = sample.navigation_s
            anchor.last_visit_travel_m = self.travel_m
            return result

        travel_since_visit = self.travel_m - anchor.last_visit_travel_m
        time_since_visit = sample.navigation_s - anchor.last_visit_s
        anchor.departed = False
        anchor.last_visit_s = sample.navigation_s
        anchor.last_visit_travel_m = self.travel_m
        if (travel_since_visit < cfg.minimum_return_travel_m or
                time_since_visit < cfg.minimum_return_seconds):
            return result

        anchor.returns += 1
        recovery_increase = max(0, sample.recoveries - anchor.first_recoveries)
        result.update(event="REVISIT", anchor_returns=anchor.returns,
                      anchor_recovery_increase=recovery_increase,
                      note=f"Separated return; travelled {travel_since_visit:.2f}m since last visit.")
        if (anchor.returns >= cfg.warning_returns and
                recovery_increase >= cfg.warning_recovery_increase):
            result.update(event="LOOP_SUSPECT",
                          note="Repeated area + recovery growth; candidate only, NOT reachability proof.")
        return result


@dataclass(frozen=True)
class GuardSettings:
    # Proposed experiment thresholds, not a learned/reachability model.
    confirmation_extra_returns: int = 1
    confirmation_extra_recoveries: int = 1
    confirmation_minimum_travel_m: float = 5.0
    novel_radius_m: float = 1.0
    novel_consecutive_samples: int = 3
    fresh_goal_max_navigation_s: float = 2.0


@dataclass
class Candidate:
    x: float
    y: float
    returns: int
    recoveries: int
    navigation_s: float
    travel_m: float
    baseline: tuple[tuple[float, float], ...]
    novel_streak: int = 0


class LoopGuard:
    """Two-stage hypothesis; returns a decision, never calls ROS."""
    def __init__(self, config: GuardSettings):
        self.config = config
        self.candidate: Candidate | None = None

    def update(self, s: Sample, decision: dict[str, Any],
               history: GoalHistory) -> dict[str, Any]:
        event = decision["event"]
        result: dict[str, Any] = {"guard_event": "", "guard_note": "",
                                 "candidate_x": "", "candidate_y": ""}
        if event in {"START", "RESET", "ANCHOR_LIMIT"}:
            was_pending = self.candidate is not None
            self.candidate = None
            if was_pending:
                result.update(guard_event="CANDIDATE_RESET",
                              guard_note="History discontinuity / capacity: discard hypothesis.")
            return result

        c = self.candidate
        if c is not None:
            result.update(candidate_x=c.x, candidate_y=c.y)
            nearest = min(math.hypot(s.x-x, s.y-y) for x, y in c.baseline)
            if nearest > self.config.novel_radius_m:
                c.novel_streak += 1
            else:
                c.novel_streak = 0
            if c.novel_streak >= self.config.novel_consecutive_samples:
                self.candidate = None
                # Fresh spatial history prevents old warnings on a later normal return.
                history._reset(s)
                result.update(guard_event="PROGRESS_RESET",
                              guard_note="Several samples enter previously unrecorded area; rebuild history.")
                return result
            same_anchor = (decision["anchor_x"] != "" and
                           math.hypot(float(decision["anchor_x"])-c.x,
                                      float(decision["anchor_y"])-c.y) < 1e-6)
            if (event in {"REVISIT", "LOOP_SUSPECT"} and same_anchor and
                    decision["anchor_returns"] >= c.returns + self.config.confirmation_extra_returns and
                    s.recoveries >= c.recoveries + self.config.confirmation_extra_recoveries and
                    history.travel_m - c.travel_m >= self.config.confirmation_minimum_travel_m):
                result.update(guard_event="LOOP_CONFIRMED",
                              guard_note="Additional separated return to the SAME warning anchor + further recovery; policy evidence, not unreachability proof.")
            return result

        if event == "LOOP_SUSPECT":
            self.candidate = Candidate(
                float(decision["anchor_x"]), float(decision["anchor_y"]),
                int(decision["anchor_returns"]), s.recoveries, s.navigation_s,
                history.travel_m, tuple((a.x, a.y) for a in history.anchors))
            result.update(guard_event="LOOP_CANDIDATE",
                          candidate_x=self.candidate.x, candidate_y=self.candidate.y,
                          guard_note="Candidate: optional immediate targeted cancel after multi-return + recovery growth; not unreachability proof.")
        return result


def valid_goal_bytes(goal_id: str) -> bytes:
    try:
        raw = bytes.fromhex(goal_id)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid UUID: refuse cancellation") from exc
    if len(raw) != 16 or not any(raw):
        raise ValueError("Zero/invalid UUID: cancel-all is prohibited")
    return raw


def fill_exact_cancel_request(request: Any, goal_id: str) -> Any:
    request.goal_info.goal_id.uuid = list(valid_goal_bytes(goal_id))
    # Nonzero UUID + ZERO timestamp targets only that UUID (official contract).
    request.goal_info.stamp.sec = 0
    request.goal_info.stamp.nanosec = 0
    return request


def auto_block_reason(enabled: bool, eligible: bool, used: bool,
                      goal_id: str, live_statuses: dict[str, int],
                      service_ready: bool) -> str:
    if not enabled:
        return "OBSERVE_ONLY"
    if not eligible:
        return "MID_GOAL_NOT_ARMED"
    if used:
        return "ONE_SHOT_ALREADY_USED"
    active = {g: st for g, st in live_statuses.items() if st in {1, 2, 3}}
    if active != {goal_id: 2}:
        return "NEED_ONE_EXECUTING_GOAL_WITH_MATCHING_UUID"
    if not service_ready:
        return "CANCEL_SERVICE_UNAVAILABLE"
    valid_goal_bytes(goal_id)
    return ""


CSV_FIELDS = [
    "wall_utc", "goal_id", "event", "action_status", "frame_id", "navigation_s",
    "pose_stamp_s", "x", "y", "yaw_rad", "distance_remaining_m", "recoveries",
    "segment", "step_m", "travel_m", "anchor_returns", "anchor_recovery_increase",
    "anchor_x", "anchor_y", "note", "guard_event", "guard_note", "candidate_x",
    "candidate_y", "auto_eligible", "mode", "cancel_code", "mission_reason",
]


def duration_seconds(value: Any) -> float:
    return float(value.sec) + float(value.nanosec) * 1e-9


def sample_is_valid(s: Sample) -> bool:
    return bool(s.frame_id and s.navigation_s >= 0 and s.pose_stamp_s >= 0 and s.recoveries >= 0
                and all(math.isfinite(v) for v in
                        (s.navigation_s, s.pose_stamp_s, s.x, s.y, s.yaw_rad)))


def replay_csv(path: Path) -> dict[str, Any]:
    histories: dict[str, GoalHistory] = {}
    guards: dict[str, LoopGuard] = {}
    start: dict[str, float] = {}
    first_candidate, first_confirmation = {}, {}
    candidate_count = confirmation_count = 0
    samples = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("navigation_s") or row.get("event") == "ACTION_STATUS":
                continue
            s = Sample(row["goal_id"], row["frame_id"], float(row["navigation_s"]),
                       float(row["pose_stamp_s"]), float(row["x"]), float(row["y"]),
                       float(row["yaw_rad"]), float(row["distance_remaining_m"]),
                       int(row["recoveries"]))
            if not sample_is_valid(s):
                continue
            if s.goal_id not in histories:
                histories[s.goal_id], guards[s.goal_id] = GoalHistory(Settings()), LoopGuard(GuardSettings())
                start[s.goal_id] = s.navigation_s
            h = histories[s.goal_id]
            decision = h.update(s)
            if decision is None:
                continue
            samples += 1
            gd = guards[s.goal_id].update(s, decision, h)
            if gd["guard_event"] == "LOOP_CANDIDATE":
                candidate_count += 1
                first_candidate.setdefault(s.goal_id, round(s.navigation_s-start[s.goal_id], 6))
            if gd["guard_event"] == "LOOP_CONFIRMED":
                confirmation_count += 1
                first_confirmation.setdefault(s.goal_id, round(s.navigation_s-start[s.goal_id], 6))
    return {"file": str(path), "samples": samples, "goal_count": len(histories),
            "candidate_events": candidate_count, "confirmation_events": confirmation_count,
            "first_candidate_after_recording_s": first_candidate,
            "first_confirmation_after_recording_s": first_confirmation,
            "scope": "Pattern logic only. No ROS service or live fresh-goal arming is exercised."}


def run_ros(args: argparse.Namespace) -> int:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import ExternalShutdownException
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from nav2_msgs.action import NavigateToPose
        from action_msgs.msg import GoalStatusArray
        from action_msgs.srv import CancelGoal
    except ImportError as exc:
        print(f"ROS 패키지 오류: {exc}\nsource /opt/ros/humble/setup.bash 후 /usr/bin/python3로 실행하세요.", file=sys.stderr)
        return 2

    namespace = "/" + args.namespace.strip("/") if args.namespace.strip("/") else ""
    action_base = namespace + "/navigate_to_pose/_action"
    config, guard_config = Settings(), GuardSettings()
    mode = "AUTO_CANCEL_FIRST_CANDIDATE_SIM_TEST" if args.auto_cancel else "OBSERVE_ONLY"
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    csv_path = out / f"navigation_loop_{tag}.csv"
    meta_path = out / f"navigation_loop_{tag}_meta.json"
    meta = {"version": VERSION, "mode": mode, "namespace": namespace,
            "feedback_topic": action_base + "/feedback", "status_topic": action_base + "/status",
            "cancel_service": action_base + "/cancel_goal" if args.auto_cancel else None,
            "settings": asdict(config), "guard_settings": asdict(guard_config),
            "time_basis": "feedback.navigation_time; no total-duration timeout",
            "notes": ["On first LOOP_CANDIDATE, opt-in mode attempts exact-Goal cancellation; unlike the conservative two-stage policy.",
                      "Heuristic simulation experiment; not proof of unreachability or a safety stop.",
                      "No velocity, goal, or parameter changes. Exact-UUID cancel is opt-in only.",
                      "One cancellation request maximum per process, without automatic retries.",
                      "Mid-goal attachment is observation only. Issue a fresh RViz Goal after READY.",
                      "CANCELED is not ABORTED; LOOP_POLICY_CANCELLED is recorded separately.",
                      "Localization estimates can be noisy. Pure rotation is not a spatial return.",
                      "Distance remaining is logged, never used alone to stop or declare success.",
                      "Successful/terminal goals suppress no-feedback WAIT notices.",
                      "Ctrl+C exits the monitor and does not issue a cancellation."]}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    rclpy.init(args=[])
    node = None
    try:
        with csv_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            handle.flush()

            class Monitor(Node):
                def __init__(self) -> None:
                    super().__init__("navigation_loop_monitor")
                    self.histories: dict[str, GoalHistory] = {}
                    self.loop_guards: dict[str, LoopGuard] = {}
                    self.eligible: dict[str, bool] = {}
                    self.statuses: dict[str, int] = {}
                    self.live_statuses: dict[str, int] = {}
                    self.last_console: dict[str, float] = {}
                    self.last_seen: dict[str, float] = {}
                    self.last_valid_sample: dict[str, Sample] = {}
                    self.last_notice = time.monotonic()
                    self.announced_idle = False
                    self.started = False
                    self.auto_used = False
                    self.requested_goal: str | None = None
                    self.request_wall = 0.0
                    self.response_seen = False
                    self.cancel_accepted = False
                    self.request_future = None
                    self.last_invalid_wall = -math.inf
                    self.client = self.create_client(CancelGoal, action_base + "/cancel_goal") if args.auto_cancel else None
                    fbq = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                                     durability=DurabilityPolicy.VOLATILE)
                    stq = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
                    self.fb_sub = self.create_subscription(NavigateToPose.Impl.FeedbackMessage,
                        action_base + "/feedback", self.on_feedback, fbq)
                    self.status_sub = self.create_subscription(GoalStatusArray,
                        action_base + "/status", self.on_status, stq)

                def write(self, row: dict[str, Any]) -> None:
                    row = dict(row)
                    row["wall_utc"] = datetime.now(timezone.utc).isoformat()
                    row.setdefault("mode", mode)
                    writer.writerow(row)
                    handle.flush()

                def event(self, goal: str, event: str, **extra: Any) -> None:
                    row: dict[str, Any] = {"goal_id": goal, "event": event,
                        "action_status": STATUS_NAMES.get(self.statuses.get(goal, 0), "UNKNOWN")}
                    row.update(extra)
                    self.write(row)

                def on_status(self, msg: Any) -> None:
                    current = {bytes(item.goal_info.goal_id.uuid).hex(): int(item.status)
                               for item in msg.status_list}
                    self.live_statuses = current
                    for gid, status in current.items():
                        prev = self.statuses.get(gid)
                        self.statuses[gid] = status
                        if prev == status:
                            continue
                        self.event(gid, "ACTION_STATUS")
                        if gid not in self.histories:
                            continue
                        print(f"[STATUS] Goal={gid[:8]} {STATUS_NAMES.get(status, status)}", flush=True)
                        if status in TERMINAL:
                            self.loop_guards[gid].candidate = None
                            self.eligible[gid] = False
                            if gid == self.requested_goal and status == 5:
                                self.event(gid, "MISSION_POLICY_STOP", mission_reason="LOOP_POLICY_CANCELLED",
                                           note="CANCELED status observed after targeted request; verify physical stop separately.")
                                print("[POLICY_STOP] 반복 탐색 정책에 따른 취소 종료 확인. 실제 정지는 화면/odom으로 확인하세요.", flush=True)
                            elif gid == self.requested_goal and status == 4:
                                print("[RACE] 취소 요청과 겹쳐 Goal이 먼저 성공했습니다. 성공 상태를 유지합니다.", flush=True)
                            if not any(self.statuses.get(g) not in TERMINAL for g in self.histories):
                                print("[IDLE] 작업 종료 — 새 Goal 대기. 완료한 Goal에는 WAIT를 반복하지 않습니다.", flush=True)
                                self.announced_idle = True

                def on_feedback(self, msg: Any) -> None:
                    gid = bytes(msg.goal_id.uuid).hex()
                    self.last_seen[gid] = time.monotonic()
                    if self.statuses.get(gid) in TERMINAL or self.statuses.get(gid) == 3:
                        return
                    f = msg.feedback
                    ps = f.current_pose
                    p, q = ps.pose.position, ps.pose.orientation
                    norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
                    if not math.isfinite(norm) or norm < 1e-9:
                        return
                    qx, qy, qz, qw = q.x/norm, q.y/norm, q.z/norm, q.w/norm
                    yaw = math.atan2(2*(qw*qz + qx*qy), 1-2*(qy*qy + qz*qz))
                    s = Sample(gid, ps.header.frame_id, duration_seconds(f.navigation_time),
                               duration_seconds(ps.header.stamp), float(p.x), float(p.y), yaw,
                               float(f.distance_remaining), int(f.number_of_recoveries))
                    if not sample_is_valid(s):
                        now = time.monotonic()
                        if now-self.last_invalid_wall > 5:
                            print("[DATA] 유효하지 않은 위치/시간: 판정 생략.", flush=True)
                            self.last_invalid_wall = now
                        return
                    self.started = True
                    self.announced_idle = False
                    if gid not in self.histories:
                        self.histories[gid] = GoalHistory(config)
                        self.loop_guards[gid] = LoopGuard(guard_config)
                        self.eligible[gid] = bool(args.auto_cancel and not self.auto_used and
                            s.navigation_s <= guard_config.fresh_goal_max_navigation_s and s.recoveries == 0)
                        label = "자동중단 대상(새 Goal)" if self.eligible[gid] else "관찰 전용"
                        print(f"[GOAL] {gid[:8]} {label}; nav={s.navigation_s:.1f}s", flush=True)
                        if args.auto_cancel and not self.eligible[gid] and not self.auto_used:
                            print("[NOT_ARMED] 중간에 연결한 Goal은 취소하지 않습니다. 기존 Goal을 취소한 뒤 새 Goal을 주세요.", flush=True)
                    h = self.histories[gid]
                    d = h.update(s)
                    if d is None:
                        return
                    self.last_valid_sample[gid] = s
                    gd = self.loop_guards[gid].update(s, d, h)
                    row = asdict(s)
                    row.update(d)
                    row.update(gd)
                    row["action_status"] = STATUS_NAMES.get(self.statuses.get(gid, 0), "UNKNOWN")
                    row["auto_eligible"] = self.eligible[gid]
                    self.write(row)
                    ge = gd["guard_event"]
                    if ge == "LOOP_CANDIDATE":
                        print(f"[LOOP_CANDIDATE] {gid[:8]} 반복 의심. --auto-cancel 활성화 시 즉시 해당 Goal 취소를 시도합니다.", flush=True)
                        self.maybe_cancel(gid)
                    elif ge == "PROGRESS_RESET":
                        print(f"[PROGRESS_RESET] {gid[:8]} 새 구역으로 진행 — 이전 반복 의심 해제.", flush=True)
                    elif ge == "CANDIDATE_RESET":
                        print(f"[CANDIDATE_RESET] {gid[:8]} 데이터 연속성/기록 한계로 의심 해제.", flush=True)
                    elif ge == "LOOP_CONFIRMED":
                        print(f"[LOOP_CONFIRMED] {gid[:8]} 추가 반복 확인 (조기 중단 실험은 이미 최초 경고 시 취소 판단).", flush=True)
                    if d["event"] == "RESET":
                        print(f"[RESET] {gid[:8]} {d['note']}", flush=True)
                    last = self.last_console.get(gid, -math.inf)
                    if s.navigation_s-last >= 5 or d["event"] in {"START", "RESET"}:
                        print(f"[LOG] Goal={gid[:8]} nav={s.navigation_s:.1f}s pose=({s.x:.2f}, {s.y:.2f}) "
                              f"remaining={s.distance_remaining_m:.2f}m recoveries={s.recoveries}", flush=True)
                        self.last_console[gid] = s.navigation_s

                def maybe_cancel(self, gid: str) -> None:
                    reason = auto_block_reason(args.auto_cancel, self.eligible.get(gid, False),
                        self.auto_used, gid, self.live_statuses,
                        bool(self.client and self.client.service_is_ready()))
                    if reason:
                        self.event(gid, "AUTO_BLOCKED", note=reason)
                        print(f"[AUTO_BLOCKED] {reason} — 취소 요청 없음.", flush=True)
                        return
                    request = fill_exact_cancel_request(CancelGoal.Request(), gid)
                    # Log intent BEFORE the only consequential operation; fail closed on disk error.
                    self.event(gid, "CANCEL_REQUESTED", mission_reason="LOOP_POLICY_REQUESTED",
                               note="Exact UUID; zero timestamp; one shot; no velocity override.")
                    self.auto_used = True
                    self.requested_goal = gid
                    self.request_wall = time.monotonic()
                    print(f"[CANCEL_REQUESTED] Goal={gid} 하나만 취소 요청. 전체 취소 아님.", flush=True)
                    try:
                        self.request_future = self.client.call_async(request)
                        self.request_future.add_done_callback(lambda future, target=gid: self.on_cancel_reply(target, future))
                    except Exception as exc:
                        self.response_seen = True
                        self.event(gid, "CANCEL_ERROR", note=str(exc))
                        print(f"[CANCEL_ERROR] {exc}; 자동 재시도 없음. 필요하면 RViz에서 취소하세요.", flush=True)

                def on_cancel_reply(self, gid: str, future: Any) -> None:
                    self.response_seen = True
                    try:
                        reply = future.result()
                        code = int(reply.return_code)
                        ids = [bytes(g.goal_id.uuid).hex() for g in reply.goals_canceling]
                        self.cancel_accepted = code == 0 and gid in ids
                        event = "CANCEL_ACCEPTED" if self.cancel_accepted else "CANCEL_REJECTED"
                        self.event(gid, event, cancel_code=code, note="goals_canceling="+",".join(ids))
                        print(f"[{event}] code={code}; 최종 STATUS CANCELED 확인이 별도로 필요합니다.", flush=True)
                    except Exception as exc:
                        self.event(gid, "CANCEL_ERROR", note=str(exc))
                        print(f"[CANCEL_ERROR] {exc}; RViz와 상태를 확인하세요.", flush=True)

                def notices(self) -> None:
                    now = time.monotonic()
                    if now-self.last_notice < 10:
                        return
                    self.last_notice = now
                    if self.requested_goal and self.statuses.get(self.requested_goal) not in TERMINAL:
                        if now-self.request_wall >= 10:
                            print("[CHECK_CANCEL] 취소 종료를 아직 확인하지 못했습니다. RViz에서 상태와 실제 움직임을 확인하세요. 자동 재시도 없음.", flush=True)
                    active = [g for g in self.histories if self.statuses.get(g) not in TERMINAL]
                    if active:
                        if all(now-self.last_seen.get(g, 0) >= 10 for g in active):
                            print("[WAIT] 실행 중 Goal의 새 Feedback 없음. 일시정지/연결을 확인하세요; 이 이유로 취소하지 않습니다.", flush=True)
                    elif not self.started and not self.announced_idle:
                        print("[IDLE] 새 RViz Goal 대기 중.", flush=True)
                        self.announced_idle = True

            node = Monitor()
            print(f"[NAVIGATION_LOOP_MONITOR] {mode}; Nav2 설정/속도명령 변경 없음", flush=True)
            print(f"[CSV] {csv_path}\n[META] {meta_path}", flush=True)
            if args.auto_cancel:
                print("[ARMED] Gazebo 실험용. 새 Goal만 대상, 취소 요청은 이 실행에서 최대 1회입니다.", flush=True)
            print("[READY] 이제 RViz에서 새 Goal을 지정하세요.", flush=True)
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.5)
                node.notices()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print(f"\n[SAVED] {csv_path}\n도구 종료. 종료 시 추가 취소는 보내지 않습니다. 주행 중이면 RViz에서 별도로 취소하세요.", flush=True)
    return 0


def base_self_test() -> int:
    config = Settings()
    def sample(t: float, x: float, y: float = 0.0, recovery: int = 0, frame: str = "map") -> Sample:
        return Sample("goal-a", frame, t, t, x, y, 0.0, 100.0-x, recovery)

    straight = GoalHistory(config)
    results = [straight.update(sample(float(t), t*0.25, recovery=t//20)) for t in range(240)]
    assert not any(r and r["event"] in {"REVISIT", "LOOP_SUSPECT"} for r in results)

    stationary = GoalHistory(config)
    results = [stationary.update(sample(float(t), 0.0, recovery=t//10)) for t in range(120)]
    assert not any(r and r["event"] == "LOOP_SUSPECT" for r in results)
    assert stationary.update(sample(119.0, 0.0, recovery=11)) is None

    def loop_trial(loops: int, with_recovery: bool) -> list[dict[str, Any]]:
        history = GoalHistory(config)
        events: list[dict[str, Any]] = []
        history.update(sample(0.0, 0.0))
        t = 0
        for lap in range(loops):
            for x in [i*0.5 for i in range(1, 11)] + [i*0.5 for i in range(9, -1, -1)]:
                t += 1
                result = history.update(sample(float(t), x, recovery=(lap+1)*2 if with_recovery else 0))
                if result:
                    events.append(result)
        return events

    normal_backtrack = loop_trial(1, True)
    assert any(r["event"] == "REVISIT" for r in normal_backtrack)
    assert not any(r["event"] == "LOOP_SUSPECT" for r in normal_backtrack)
    repetitive = loop_trial(4, True)
    assert any(r["event"] == "LOOP_SUSPECT" for r in repetitive)
    assert not any(r["event"] == "LOOP_SUSPECT" for r in loop_trial(4, False))

    reset = GoalHistory(config)
    reset.update(sample(0.0, 0.0))
    assert reset.update(sample(1.0, 20.0))["event"] == "RESET"
    assert reset.update(sample(2.0, 20.0, frame="odom"))["event"] == "RESET"
    assert reset.update(sample(0.0, 20.0, frame="odom"))["event"] == "RESET"
    assert reset.update(sample(20.0, 20.0, frame="odom"))["event"] == "RESET"
    separate_goal = GoalHistory(config)
    assert separate_goal.update(sample(100.0, 0.0))["event"] == "START"
    print("SELF-TEST PASS: straight motion, stationary/duplicate samples, normal backtracking, "
          "repeated returns, recovery gating, clock/frame/jump/gap reset, isolated histories.")
    print("ROS2 integration is not exercised by this self-test.")
    return 0


def self_test() -> int:
    from types import SimpleNamespace
    base_self_test()
    cfg = Settings()
    gc = GuardSettings()
    gid = "12345678123456781234567812345678"
    def s(t: float, x: float, y: float = 0., recovery: int = 0) -> Sample:
        return Sample(gid, "map", t, t+100, x, y, t/10, 100-x, recovery)
    def run_path(samples: list[Sample]) -> tuple[list[str], GoalHistory, LoopGuard]:
        h, g, events = GoalHistory(cfg), LoopGuard(gc), []
        for sample in samples:
            d = h.update(sample)
            if d is not None:
                events.append(g.update(sample, d, h)["guard_event"])
        return events, h, g
    def loop_samples(laps: int, recovery: bool = True) -> list[Sample]:
        samples = [s(0., 0.)]
        t = 0
        for lap in range(laps):
            for x in [i*.5 for i in range(1, 11)]+[i*.5 for i in range(9,-1,-1)]:
                t += 1
                samples.append(s(float(t), x, recovery=lap+1 if recovery else 0))
        return samples
    events, _, _ = run_path([s(float(t), t*.5, recovery=t//20) for t in range(500)])
    assert "LOOP_CONFIRMED" not in events
    events, _, _ = run_path([s(float(t), 0., recovery=t//10) for t in range(120)])
    assert "LOOP_CANDIDATE" not in events
    events, _, _ = run_path(loop_samples(1))
    assert "LOOP_CANDIDATE" not in events
    events, _, _ = run_path(loop_samples(8, False))
    assert "LOOP_CONFIRMED" not in events
    events, _, _ = run_path(loop_samples(8))
    assert "LOOP_CANDIDATE" in events and "LOOP_CONFIRMED" in events
    assert events.index("LOOP_CONFIRMED") > events.index("LOOP_CANDIDATE")
    # Candidate-only policy: run ROS cancellation at the earliest candidate,
    # without needing LOOP_CONFIRMED. This source test checks the event hook.
    assert 'self.maybe_cancel(gid)' in Path(__file__).read_text(encoding='utf-8').split('if ge == "LOOP_CANDIDATE":',1)[1].split('elif ge == "PROGRESS_RESET":',1)[0]

    # A single first warning cannot confirm. A different anchor cannot confirm.
    h, g = GoalHistory(cfg), LoopGuard(gc)
    h.update(s(0, 0))
    d = {"event": "LOOP_SUSPECT", "anchor_x": 0., "anchor_y": 0., "anchor_returns": 3}
    assert g.update(s(20, 0, recovery=2), d, h)["guard_event"] == "LOOP_CANDIDATE"
    h.travel_m = 10
    d2 = dict(d, anchor_returns=4, anchor_x=2.)
    assert g.update(s(40, 2, recovery=3), d2, h)["guard_event"] != "LOOP_CONFIRMED"
    d2["anchor_x"] = 0.
    assert g.update(s(41, 0, recovery=2), d2, h)["guard_event"] != "LOOP_CONFIRMED"
    assert g.update(s(42, 0, recovery=3), d2, h)["guard_event"] == "LOOP_CONFIRMED"
    # Continuing into three new spatial samples clears the pending candidate.
    for t in range(43, 46):
        r = g.update(s(t, t-40.), {"event": "RECORDING", "anchor_x": "", "anchor_y": ""}, h)
    assert r["guard_event"] == "PROGRESS_RESET" and g.candidate is None
    # Reset / gap / coordinate change discards pending evidence.
    g.update(s(50, 0, recovery=4), d, h)
    assert g.candidate is not None
    g.update(s(80, 0, recovery=4), {"event": "RESET"}, h)
    assert g.candidate is None
    h2 = GoalHistory(cfg)
    h2.update(s(0,0))
    stale = Sample(gid, "map", 1., 100., 0., 0., 0., 0., 0)
    assert h2.update(stale)["event"] == "RESET"
    invalid = s(1., float('nan'))
    assert not sample_is_valid(invalid)

    assert auto_block_reason(False, True, False, gid, {gid:2}, True) == "OBSERVE_ONLY"
    assert auto_block_reason(True, False, False, gid, {gid:2}, True) == "MID_GOAL_NOT_ARMED"
    assert auto_block_reason(True, True, True, gid, {gid:2}, True) == "ONE_SHOT_ALREADY_USED"
    for live in ({}, {gid:3}, {gid:4}, {gid:5}, {gid:6}, {gid:2, 'other':1}):
        assert auto_block_reason(True, True, False, gid, live, True)
    assert auto_block_reason(True, True, False, gid, {gid:2}, False) == "CANCEL_SERVICE_UNAVAILABLE"
    assert auto_block_reason(True, True, False, gid, {gid:2}, True) == ""
    req = SimpleNamespace(goal_info=SimpleNamespace(goal_id=SimpleNamespace(uuid=[]),
                           stamp=SimpleNamespace(sec=42, nanosec=10)))
    fill_exact_cancel_request(req, gid)
    assert bytes(req.goal_info.goal_id.uuid).hex() == gid
    assert req.goal_info.stamp.sec == req.goal_info.stamp.nanosec == 0
    for bad in ('0'*32, '', 'xx', '12'):
        try:
            valid_goal_bytes(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsafe UUID accepted')
    print("MONITOR SELF-TEST PASS: candidate-first opt-in hook, two-stage detection retained, same-anchor and recovery gates, new-area reset, "
          "data-continuity reset, observe/default/mid-goal/terminal/multiple-goal/one-shot protection, "
          "exact nonzero UUID with zero timestamp.")
    print("These tests do NOT exercise a real ROS2 server, robot, or physical stopping distance.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--namespace", default="/j100_0001")
    p.add_argument("--output-dir", default=str(Path.home()/"navigation_monitor_logs"))
    p.add_argument("--auto-cancel", action="store_true", help="Opt-in Gazebo experiment: one UUID-specific cancellation maximum per process.")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--replay-csv", type=Path, help="Offline detector-only replay of a previously recorded navigation CSV.")
    args = p.parse_args()
    if args.self_test:
        return self_test()
    if args.replay_csv:
        print(json.dumps(replay_csv(args.replay_csv), ensure_ascii=False, indent=2))
        return 0
    return run_ros(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"[ERROR] 실행/저장 오류: {exc}. 도구 중단은 로봇 정지가 아닙니다. RViz 상태를 확인하세요.", file=sys.stderr)
        raise SystemExit(2)
