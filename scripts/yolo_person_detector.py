#!/usr/bin/env python3
"""ROS 2 YOLO person detection, optional preview, confirmation and snapshots.

Default: confirm continuous detections for one second and save one JPEG.
    python scripts/yolo_person_detector.py --show
    python scripts/yolo_person_detector.py --show --no-save
    python scripts/yolo_person_detector.py --show --no-confirm

Timing uses time.monotonic(), NOT a frame count or simulation /clock.
Continuity means matching boxes in successive successfully processed images.
A missed detection drops that track immediately; a reappearance is a new track.
Box IoU is only a lightweight association heuristic, not person identification.
Only a CANDIDATE -> CONFIRMED transition triggers a single save attempt.
Disabling confirmation also disables snapshots; per-frame detection remains.
This node publishes images and detections and does not command robot motion.
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO

CAMERA_TOPIC = "/j100_0001/sensors/camera_0/color/image"
ANNOTATED_TOPIC = "/person_detector/annotated_image"
DETECTIONS_TOPIC = "/person_detector/person_detections"
CONFIRM_SECONDS = 1.0
MATCH_IOU = 0.30  # Box association only; not another confidence/confirmation gate.
BBox = Tuple[float, float, float, float]


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union for two pixel-space xyxy boxes."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass
class DetectionTrack:
    track_id: int
    bbox: BBox
    first_seen: float
    last_seen: float
    confirmed: bool = False
    snapshot_attempted: bool = False
    snapshot_path: Optional[str] = None
    snapshot_error: Optional[str] = None

    @property
    def observed_seconds(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)


class ContinuousConfirmation:
    """One-to-one box association across adjacent processed frames only.

    No missed-frame grace period or retained lost identities. A current frame
    without a matching detection immediately removes the previous track.
    update() must be called once per processed image, including empty results.
    """

    def __init__(self, confirm_seconds: float = CONFIRM_SECONDS):
        if not math.isfinite(confirm_seconds) or confirm_seconds <= 0.0:
            raise ValueError("confirm_seconds must be finite and positive")
        self.confirm_seconds = confirm_seconds
        self.tracks: Dict[int, DetectionTrack] = {}
        self.next_id = 1

    def reset(self):
        self.tracks.clear()

    def update(self, boxes: Sequence[BBox], now: float):
        # A detection and a previous track may each be used at most once.
        pairs = []
        for track_id, track in self.tracks.items():
            for index, box in enumerate(boxes):
                overlap = box_iou(track.bbox, box)
                if overlap >= MATCH_IOU:
                    pairs.append((overlap, track_id, index))
        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
        assignments = {}
        used_tracks = set()
        for _, track_id, index in pairs:
            if track_id not in used_tracks and index not in assignments:
                assignments[index] = track_id
                used_tracks.add(track_id)

        active = {}
        ordered = []
        newly_confirmed = []
        for index, box in enumerate(boxes):
            track_id = assignments.get(index)
            if track_id is None:
                track = DetectionTrack(self.next_id, tuple(box), now, now)
                self.next_id += 1
            else:
                track = self.tracks[track_id]
                track.bbox = tuple(box)
                track.last_seen = now
            if not track.confirmed and track.observed_seconds >= self.confirm_seconds:
                track.confirmed = True
                newly_confirmed.append(track)
            active[track.track_id] = track
            ordered.append(track)
        self.tracks = active  # Unmatched tracks, including confirmed ones, are dropped.
        return ordered, newly_confirmed


def draw_status(frame, persons, confirm_seconds: float):
    """Draw CANDIDATE / CONFIRMED boxes on a copy of the BGR camera image."""
    overlay = frame.copy()
    height, width = overlay.shape[:2]
    for person in persons:
        x1, y1, x2, y2 = (int(round(v)) for v in person["bbox_xyxy_pixels"])
        x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
        y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
        confirmed = person["status"] == "CONFIRMED"
        color = (0, 220, 0) if confirmed else (0, 220, 255)  # BGR: green / yellow.
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        if confirmed:
            duration_label = f"{person['observed_seconds']:.2f}s"
        else:
            # Do not round 0.999s up to a misleading '1.00s' before confirmation.
            shown = math.floor(person["observed_seconds"] * 100.0) / 100.0
            duration_label = f"{shown:.2f}/{confirm_seconds:g}s"
        label = (
            f"#{person['track_id']} person {person['confidence']:.2f} "
            f"{person['status']} {duration_label}"
        )
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
        )
        tx = max(0, min(x1, width - text_width - 2))
        ty = max(text_height + 2, y1 - 7)
        cv2.putText(overlay, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
    confirmed_count = sum(p["status"] == "CONFIRMED" for p in persons)
    text = f"Persons: {len(persons)} | confirmed: {confirmed_count}"
    cv2.putText(overlay, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(overlay, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (0, 255, 255), 2, cv2.LINE_AA)
    return overlay


class PersonDetector(Node):
    def __init__(self, model_file: str, confidence: float, image_size: int, show: bool,
                 confirm_seconds: float, save_dir: Path, *, confirm: bool = True,
                 save: bool = True):
        if save and not confirm:
            raise ValueError("Snapshots require continuous confirmation")
        super().__init__("yolo_person_detector")
        self.bridge = CvBridge()
        self.confidence = confidence
        self.image_size = image_size
        self.processed = 0
        self.show = show
        self.exit_requested = False
        self.window_name = "YOLO Person Detection"
        self.confirmation = ContinuousConfirmation(confirm_seconds) if confirm else None
        self.save_dir = None
        if save:
            session = datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
            self.save_dir = save_dir.expanduser().resolve() / session
            self.save_dir.mkdir(parents=True, exist_ok=False)

        self.get_logger().info(f"Loading YOLO model: {model_file}")
        self.model = YOLO(model_file)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Inference device: {self.device}")

        self.subscriber = self.create_subscription(
            Image, CAMERA_TOPIC, self.handle_image, qos_profile_sensor_data
        )
        self.annotated_publisher = self.create_publisher(Image, ANNOTATED_TOPIC, 10)
        self.detections_publisher = self.create_publisher(String, DETECTIONS_TOPIC, 10)
        self.get_logger().info(f"Waiting for RGB images: {CAMERA_TOPIC}")
        self.get_logger().info(f"Annotated image: {ANNOTATED_TOPIC}")
        self.get_logger().info(f"Detections (JSON String): {DETECTIONS_TOPIC}")
        if self.confirmation is not None:
            self.get_logger().info(
                f"Confirmation: {confirm_seconds:g}s continuous processed-frame detections; "
                "wall-monotonic time, reset on miss"
            )
        else:
            self.get_logger().info("Confirmation disabled; per-frame detections only.")
        self.get_logger().info(f"Snapshots: {self.save_dir if self.save_dir is not None else 'disabled'}")
        if self.show:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self.get_logger().info("OpenCV preview enabled. Press q in the image window to quit.")

    def save_snapshot(self, track: DetectionTrack, overlay):
        """One write attempt per confirmation; never report a failed write as saved."""
        if self.save_dir is None or track.snapshot_attempted:
            return
        track.snapshot_attempted = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination = self.save_dir / f"victim_{track.track_id:04d}_{timestamp}.jpg"
        # Full annotated frame; the footer identifies which track triggered this photo.
        photo = overlay.copy()
        caption = f"SNAPSHOT: track #{track.track_id} | {timestamp}"
        cv2.putText(photo, caption, (10, photo.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(photo, caption, (10, photo.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        try:
            if not cv2.imwrite(str(destination), photo):
                raise OSError("cv2.imwrite returned False")
        except Exception as exc:
            track.snapshot_error = str(exc)
            self.get_logger().error(
                f"[PHOTO_SAVE_FAILED] id={track.track_id}; {destination}; {exc}"
            )
            return
        track.snapshot_path = str(destination)
        self.get_logger().info(f"[PHOTO_SAVED] id={track.track_id}; {destination}")

    def handle_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            prediction = self.model.predict(
                source=frame,
                classes=[0],
                conf=self.confidence,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )[0]
            persons = []
            boxes = []
            for box in prediction.boxes:
                xyxy = tuple(float(v) for v in box.xyxy[0].cpu().tolist())
                boxes.append(xyxy)
                persons.append({
                    "class": "person",
                    "confidence": round(float(box.conf[0].item()), 4),
                    "bbox_xyxy_pixels": [round(v, 2) for v in xyxy],
                })

            if self.confirmation is not None:
                # Start on the first inference result, not during initial model loading.
                tracks, newly_confirmed = self.confirmation.update(boxes, time.monotonic())
                for person, track in zip(persons, tracks):
                    person.update({
                        "track_id": track.track_id,
                        "status": "CONFIRMED" if track.confirmed else "CANDIDATE",
                        "observed_seconds": track.observed_seconds,
                    })
                overlay = draw_status(frame, persons, self.confirmation.confirm_seconds)
                for track in newly_confirmed:
                    self.get_logger().info(
                        f"[CONFIRMED] id={track.track_id}; "
                        f"continuous={track.observed_seconds:.3f}s"
                    )
                    self.save_snapshot(track, overlay)
                for person, track in zip(persons, tracks):
                    person.update({
                        "snapshot_saved": track.snapshot_path is not None,
                        "snapshot_path": track.snapshot_path,
                        "snapshot_error": track.snapshot_error,
                    })
            else:
                overlay = prediction.plot()
                cv2.putText(overlay, f"Persons: {len(persons)}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            annotated = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            annotated.header = msg.header
            self.annotated_publisher.publish(annotated)
            if self.show:
                cv2.imshow(self.window_name, overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.exit_requested = True

            confirmed_count = sum(p.get("status") == "CONFIRMED" for p in persons)
            payload = {
                "frame_id": msg.header.frame_id,
                "stamp_sec": msg.header.stamp.sec,
                "stamp_nanosec": msg.header.stamp.nanosec,
                "image_width": int(msg.width),
                "image_height": int(msg.height),
                "person_count": len(persons),  # Original meaning: raw per-frame detections.
                "detections": persons,
            }
            if self.confirmation is not None:
                payload.update(
                    confirmed_count=confirmed_count,
                    candidate_count=len(persons) - confirmed_count,
                    confirmation_seconds=self.confirmation.confirm_seconds,
                )
            out = String()
            out.data = json.dumps(payload, separators=(",", ":"))
            self.detections_publisher.publish(out)
            self.processed += 1
            if self.processed == 1 or self.processed % 30 == 0:
                confidence_values = ", ".join(f"{p['confidence']:.2f}" for p in persons) or "none"
                self.get_logger().info(
                    f"frames={self.processed}, persons={len(persons)}, "
                    f"confirmed={confirmed_count}, confidence=[{confidence_values}]"
                )
        except Exception as exc:
            # An unprocessed/error frame cannot establish continuous detection.
            if self.confirmation is not None:
                self.confirmation.reset()
            self.get_logger().error(f"Image processing failed: {exc}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained YOLO detect model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0-1)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--show", action="store_true", help="Show the OpenCV window; q quits")
    parser.add_argument("--confirm", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable continuous confirmation (default: enabled)")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=None,
                        help="Save one JPEG per confirmed track (default: enabled with confirmation)")
    parser.add_argument("--confirm-seconds", type=float, default=CONFIRM_SECONDS,
                        help="Continuous detection duration in real elapsed seconds (default: 1)")
    parser.add_argument("--save-dir", type=Path, default=Path("logs/victim_snapshots"),
                        help="Parent folder for session snapshots (relative to working directory)")
    args = parser.parse_args(argv)
    if args.save is None:
        args.save = args.confirm
    if args.save and not args.confirm:
        parser.error("--save requires --confirm; use --no-confirm without --save for basic detection")
    if not (0 < args.conf <= 1):
        parser.error("--conf must be in (0, 1]")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    if not math.isfinite(args.confirm_seconds) or args.confirm_seconds <= 0.0:
        parser.error("--confirm-seconds must be finite and positive")
    if args.show and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        parser.error("--show needs a graphical desktop (DISPLAY or WAYLAND_DISPLAY)")
    return args


def main():
    args = parse_args()
    rclpy.init(args=[])
    node = None
    try:
        node = PersonDetector(args.model, args.conf, args.imgsz, args.show,
                              args.confirm_seconds, args.save_dir,
                              confirm=args.confirm, save=args.save)
        while rclpy.ok() and not node.exit_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if node.show:
                cv2.destroyAllWindows()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
