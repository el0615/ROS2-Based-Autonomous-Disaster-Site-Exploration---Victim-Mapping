"""Offline detector regressions: real OpenCV images, mocked ROS and inference."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np


class FakeNode:
    def __init__(self, name):
        self.node_name = name
        self.logger = Mock()
        self.publishers = {}

    def get_logger(self):
        return self.logger

    def create_subscription(self, *args):
        self.subscription_args = args
        return object()

    def create_publisher(self, message_type, topic, depth):
        publisher = Mock()
        self.publishers[topic] = publisher
        return publisher


class FakeBridge:
    def imgmsg_to_cv2(self, msg, desired_encoding):
        return np.zeros((msg.height, msg.width, 3), dtype=np.uint8)

    def cv2_to_imgmsg(self, image, encoding):
        return types.SimpleNamespace(image=image.copy(), encoding=encoding)


def load_detector():
    modules = {name: types.ModuleType(name) for name in (
        'rclpy', 'rclpy.node', 'rclpy.qos', 'cv_bridge', 'sensor_msgs',
        'sensor_msgs.msg', 'std_msgs', 'std_msgs.msg', 'torch', 'ultralytics')}
    modules['rclpy.node'].Node = FakeNode
    modules['rclpy.qos'].qos_profile_sensor_data = object()
    modules['cv_bridge'].CvBridge = FakeBridge
    modules['sensor_msgs.msg'].Image = type('Image', (), {})
    modules['std_msgs.msg'].String = type('String', (), {})
    modules['torch'].cuda = types.SimpleNamespace(is_available=lambda: False)
    modules['ultralytics'].YOLO = Mock()
    path = Path(__file__).resolve().parents[1] / 'scripts/yolo_person_detector.py'
    spec = importlib.util.spec_from_file_location('_tested_person_detector', path)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve annotations using the module's registered name.
    with patch.dict(sys.modules, {**modules, spec.name: module}):
        spec.loader.exec_module(module)
    return module


detector = load_detector()
BOX = (20., 40., 80., 110.)


class ConfirmationTests(unittest.TestCase):
    def test_threshold_and_single_transition(self):
        tracker = detector.ContinuousConfirmation()
        for now in (10., 10.2, 10.999):
            tracks, confirmed = tracker.update([BOX], now)
            self.assertFalse(confirmed)
            self.assertFalse(tracks[0].confirmed)
        tracks, confirmed = tracker.update([BOX], 11.)
        self.assertEqual(confirmed, tracks)
        self.assertEqual(tracker.update([BOX], 12.)[1], [])

    def test_missed_detection_restarts_confirmation(self):
        tracker = detector.ContinuousConfirmation(.5)
        first = tracker.update([BOX], 0.)[0][0]
        tracker.update([], .4)
        tracks, confirmed = tracker.update([BOX], .6)
        self.assertNotEqual(tracks[0].track_id, first.track_id)
        self.assertFalse(confirmed)
        self.assertEqual(tracks[0].observed_seconds, 0.)

    def test_two_boxes_cannot_share_a_previous_track(self):
        tracker = detector.ContinuousConfirmation()
        tracker.update([BOX], 0.)
        tracks, confirmed = tracker.update([BOX, BOX], 1.)
        self.assertEqual(len({t.track_id for t in tracks}), 2)
        self.assertEqual(len(confirmed), 1)


class DetectorTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name) / 'photos'
        self.model = Mock()
        model_patch = patch.object(detector, 'YOLO', return_value=self.model)
        model_patch.start()
        self.addCleanup(model_patch.stop)
        self.msg = types.SimpleNamespace(width=200, height=150,
            header=types.SimpleNamespace(frame_id='camera_optical',
                stamp=types.SimpleNamespace(sec=42, nanosec=123)))

    def node(self, **options):
        return detector.PersonDetector('local.pt', .25, 640, False, 1., self.directory, **options)

    def frame(self, node, now, boxes=(BOX,)):
        predictions = []
        for box in boxes:
            xyxy = Mock(); xyxy.cpu.return_value.tolist.return_value = box
            confidence = Mock(); confidence.item.return_value = .9
            predictions.append(types.SimpleNamespace(xyxy=[xyxy], conf=[confidence]))
        result = Mock(boxes=predictions)
        result.plot.return_value = np.zeros((150, 200, 3), np.uint8)
        self.model.predict.return_value = [result]
        with patch.object(detector.time, 'monotonic', return_value=now):
            node.handle_image(self.msg)
        return json.loads(node.detections_publisher.publish.call_args.args[0].data)

    def test_default_saves_one_real_jpeg_and_keeps_camera_header(self):
        node = self.node()
        self.assertEqual(self.frame(node, 0.)['candidate_count'], 1)
        self.assertEqual(list(node.save_dir.glob('*.jpg')), [])
        report = self.frame(node, 1.)
        person = report['detections'][0]
        self.assertTrue(person['snapshot_saved'])
        self.assertEqual(report['confirmed_count'], 1)
        self.assertIsNotNone(cv2.imread(person['snapshot_path']))
        self.frame(node, 2.)
        self.assertEqual(len(list(node.save_dir.glob('*.jpg'))), 1)
        self.assertIs(node.annotated_publisher.publish.call_args.args[0].header, self.msg.header)
        self.assertEqual(node.node_name, 'yolo_person_detector')
        self.assertEqual(set(node.publishers), {
            '/person_detector/annotated_image', '/person_detector/person_detections'})

    def test_no_save_confirms_without_creating_a_directory(self):
        node = self.node(save=False)
        self.frame(node, 0.)
        report = self.frame(node, 1.)
        self.assertEqual(report['confirmed_count'], 1)
        self.assertFalse(report['detections'][0]['snapshot_saved'])
        self.assertFalse(self.directory.exists())

    def test_basic_mode_preserves_per_frame_payload_without_tracking_or_files(self):
        node = self.node(confirm=False, save=False)
        report = self.frame(node, 0.)
        self.assertEqual(set(report), {'frame_id', 'stamp_sec', 'stamp_nanosec',
            'image_width', 'image_height', 'person_count', 'detections'})
        self.assertEqual(set(report['detections'][0]), {'class', 'confidence', 'bbox_xyxy_pixels'})
        self.assertEqual(report['person_count'], 1)
        self.assertEqual(self.frame(node, 1., [])['person_count'], 0)
        self.assertIsNone(node.confirmation)
        self.assertFalse(self.directory.exists())

    def test_failed_snapshot_is_reported_once(self):
        node = self.node()
        self.frame(node, 0.)
        with patch.object(detector.cv2, 'imwrite', return_value=False) as write:
            report = self.frame(node, 1.)
            self.frame(node, 2.)
        self.assertEqual(write.call_count, 1)
        self.assertFalse(report['detections'][0]['snapshot_saved'])
        self.assertIn('returned False', report['detections'][0]['snapshot_error'])

    def test_processing_error_resets_continuity(self):
        node = self.node(save=False)
        self.frame(node, 0.)
        self.model.predict.side_effect = RuntimeError('inference failed')
        node.handle_image(self.msg)
        self.assertEqual(node.confirmation.tracks, {})
        self.model.predict.side_effect = None
        report = self.frame(node, 2.)
        self.assertEqual(report['confirmed_count'], 0)
        self.assertEqual(report['detections'][0]['observed_seconds'], 0.)

    def test_preview_q_requests_exit(self):
        node = self.node(confirm=False, save=False)
        node.show = True
        with patch.object(detector.cv2, 'imshow') as show, patch.object(detector.cv2, 'waitKey', return_value=ord('q')):
            self.frame(node, 0.)
        self.assertTrue(node.exit_requested)
        show.assert_called_once()

    def test_independent_runs_use_separate_snapshot_folders(self):
        first, second = self.node(), self.node()
        self.assertNotEqual(first.save_dir, second.save_dir)
        for node in (first, second):
            self.frame(node, 0.); self.frame(node, 1.)
            self.assertEqual(len(list(node.save_dir.glob('*.jpg'))), 1)


class OptionsTests(unittest.TestCase):
    def test_defaults_and_feature_switches(self):
        args = detector.parse_args([])
        self.assertTrue(args.confirm and args.save)
        self.assertFalse(args.show)
        self.assertEqual(args.save_dir, Path('logs/victim_snapshots'))
        self.assertEqual(args.confirm_seconds, 1.)
        args = detector.parse_args(['--no-confirm'])
        self.assertFalse(args.confirm or args.save)
        args = detector.parse_args(['--no-save', '--confirm-seconds', '.5'])
        self.assertTrue(args.confirm)
        self.assertFalse(args.save)
        self.assertEqual(args.confirm_seconds, .5)

    def test_invalid_or_incompatible_options_fail_before_ros(self):
        for argv in (['--no-confirm', '--save'], ['--confirm-seconds', 'nan'],
                     ['--confirm-seconds', '0'], ['--conf', '2'], ['--imgsz', '0']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                detector.parse_args(argv)

    def test_preview_requires_display(self):
        with patch.dict(detector.os.environ, {}, clear=True), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                detector.parse_args(['--show'])


if __name__ == '__main__':
    unittest.main()
