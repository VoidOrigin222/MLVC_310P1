"""Regression checks for published-feed status and measured bitrate units."""
import importlib
import io
import json
import unittest
from collections import deque
from unittest.mock import Mock, patch


with patch('socket.socket', side_effect=OSError('No UDP listener during tests')):
    ui = importlib.import_module('web_ui')


class StreamStatusTests(unittest.TestCase):
    def setUp(self):
        with patch.object(ui.Controller, '_ensure_mlvc_stats_socket'):
            self.controller = ui.Controller()
        self.controller.cfg['mode'] = 'same_bandwidth'
        self.addCleanup(self.controller.close)

    def status(self, items, now):
        payload = io.BytesIO(json.dumps({'items': items}).encode())
        with patch.object(ui, 'urlopen', return_value=payload), patch.object(ui.time, 'monotonic', return_value=now):
            return self.controller.status()

    def test_only_ready_paths_are_published(self):
        result = self.status([{'name': 'camera-original', 'ready': False}, {'name': 'mlvc', 'ready': True}], 100)
        self.assertFalse(result['feeds']['original'])
        self.assertTrue(result['feeds']['mlvc'])
        self.assertEqual(result['paths'], ['mlvc'])

    def test_h264_needs_two_samples_and_uses_decimal_kbps(self):
        first = self.status([{'name': 'camera-h264', 'ready': True, 'bytesReceived': 1000}], 100)
        self.assertFalse(first['h264_actual'])
        self.assertEqual(first['metrics']['h264']['kbps'], 0)
        measured = self.status([{'name': 'camera-h264', 'ready': True, 'bytesReceived': 101000}], 101)
        self.assertTrue(measured['h264_actual'])
        self.assertAlmostEqual(measured['metrics']['h264']['kbps'], 800)
        self.assertAlmostEqual(measured['metrics']['h264']['kbs'], 100000 / 1024)
        self.assertAlmostEqual(measured['metrics']['h264']['bpp'], 800000 / (1920 * 1080 * 30))

    def test_feed_disappearing_clears_counter_baseline(self):
        self.status([{'name': 'camera-h264', 'ready': True, 'bytesReceived': 1000}], 100)
        self.status([{'name': 'camera-h264', 'ready': True, 'bytesReceived': 101000}], 101)
        absent = self.status([], 102)
        self.assertFalse(absent['h264_actual'])
        self.assertEqual(absent['metrics']['h264']['kbps'], 0)
        restored = self.status([{'name': 'camera-h264', 'ready': True, 'bytesReceived': 2000}], 103)
        self.assertFalse(restored['h264_actual'])

    def test_counter_reset_and_stale_window(self):
        samples = deque([(100, 1000), (101, 101000)])
        self.assertEqual(self.controller._window_rate(samples, 102), 100000)
        self.assertEqual(self.controller._window_rate(samples, 114), 0)
        self.assertTrue(self.controller._record_counter_sample(samples, 115, 500))
        self.assertEqual(list(samples), [(115, 500)])

    def test_original_theoretical_units(self):
        metric = self.status([], 100)['metrics']['original']
        self.assertEqual(metric['kbps'], 1920 * 1080 * 30 * 24 / 1000)
        self.assertEqual(metric['kbs'], 1920 * 1080 * 30 * 3 / 1024)

    def test_bitrate_target_uses_codec_bytes_not_wire_or_pacing_budget(self):
        self.controller._mlvc_media_samples = deque([(100, 0), (112, 180000)])
        self.controller._mlvc_wire_samples = deque([(100, 0), (112, 225000)])
        self.controller._mlvc_stats_latest = {'received_at': 112}
        self.controller.cfg['mlvc_kbps'] = 2000
        self.assertAlmostEqual(self.controller._bitrate_target(112), 120)
        self.assertIsNone(self.controller._bitrate_target(125))

    def test_no_fallback_encoder_without_mlvc_statistics(self):
        self.controller._h264_requested = True
        with patch.object(self.controller, '_launch_h264') as launch:
            self.controller._sync_bitrate(112, True)
        launch.assert_not_called()
        self.assertEqual(self.controller._h264_match_state, 'waiting')

    def test_short_coalesced_statistics_cannot_set_bitrate(self):
        self.controller._mlvc_media_samples = deque([(100, 0), (100.001, 180000)])
        self.controller._mlvc_stats_latest = {'received_at': 100.001}
        self.assertIsNone(self.controller._bitrate_target(100.001))

    def test_initial_encoder_follows_measured_bitrate_below_old_floor(self):
        self.controller._h264_requested = True
        with patch.object(self.controller, '_bitrate_target', return_value=70), \
                patch.object(self.controller, '_launch_h264') as launch:
            self.controller._sync_bitrate(112, True)
        launch.assert_called_once_with(70)

    def test_running_comparison_keeps_its_budget_despite_mlvc_variation(self):
        self.controller._h264_requested = True
        self.controller.h264_encoder = Mock()
        self.controller._h264_target_kbps = 100
        self.controller._h264_rate_kbps = 100
        self.controller._h264_last_adjustment = 100
        with patch.object(self.controller, '_bitrate_target', return_value=50), \
                patch.object(self.controller, '_launch_h264') as launch:
            self.controller._sync_bitrate(200, True)
            launch.assert_not_called()
            self.assertEqual(self.controller._h264_target_kbps, 100)
            self.assertEqual(self.controller._h264_match_state, 'running')

    def test_explicit_stop_never_restarts_from_statistics(self):
        self.controller._h264_requested = True
        self.controller.h264_encoder = Mock()
        self.controller.stop()
        with patch.object(self.controller, '_bitrate_target', return_value=100), \
                patch.object(self.controller, '_launch_h264') as launch:
            self.controller._sync_bitrate(200, True)
            launch.assert_not_called()

    def test_encoder_retry_waits_and_reuses_startup_budget(self):
        self.controller._h264_requested = True
        self.controller._h264_target_kbps = 100
        self.controller._h264_rate_kbps = 100
        self.controller._h264_last_adjustment = 100
        with patch.object(self.controller, '_bitrate_target', return_value=200), \
                patch.object(self.controller, '_launch_h264') as launch:
            self.controller._sync_bitrate(112, True)
            launch.assert_not_called()
            self.controller._sync_bitrate(131, True)
            launch.assert_called_once_with(100)

    def test_quality_mode_accepts_qp_and_rejects_old_psnr_mode(self):
        self.controller.configure({'mode': 'same_quality', 'h264_qp': 37})
        self.assertEqual(self.controller.cfg['mode'], 'same_quality')
        self.assertEqual(self.controller.cfg['h264_qp'], 37)
        with self.assertRaises(ValueError):
            self.controller.configure({'mode': 'same_psnr'})
        self.assertEqual(self.controller.cfg['mode'], 'same_quality')

    def test_quality_encoder_uses_qp_without_crf_or_bitrate_limits(self):
        process = Mock()
        process.poll.return_value = None
        with patch.object(ui.subprocess, 'Popen', return_value=process) as spawn, \
                patch('builtins.open', return_value=io.BytesIO()):
            self.controller._launch_h264(qp=37)
        args = spawn.call_args[0][0]
        self.assertEqual(args[args.index('-qp') + 1], '37')
        for option in ('-crf', '-b:v', '-maxrate', '-minrate', '-bufsize', '-x264-params'):
            self.assertNotIn(option, args)

    def test_invalid_qp_does_not_change_config_or_stop_running_encoder(self):
        process = Mock()
        self.controller.h264_encoder = process
        for qp in (-1, 52, 25.5, True, '40', None):
            with self.subTest(qp=qp), self.assertRaises(ValueError):
                self.controller.start({'mode': 'same_quality', 'h264_qp': qp})
            self.assertEqual(self.controller.cfg['mode'], 'same_bandwidth')
            self.assertEqual(self.controller.cfg['h264_qp'], 40)
            process.terminate.assert_not_called()
        self.controller.h264_encoder = None

    def test_quality_starts_without_mlvc_samples_and_keeps_qp_during_motion(self):
        self.controller.cfg.update(mode='same_quality', h264_qp=37)
        self.controller._h264_requested = True
        self.controller._mlvc_stats_latest = None
        with patch.object(self.controller, '_launch_h264') as launch:
            self.controller._sync_bitrate(112, True)
            launch.assert_called_once_with(qp=37)
            self.controller.h264_encoder = Mock()
            self.controller._mlvc_media_samples = deque([(100, 0), (112, 2000000)])
            self.controller._sync_bitrate(120, True)
            launch.assert_called_once_with(qp=37)

    def test_quality_waits_for_source_and_stop_prevents_restart(self):
        self.controller.cfg['mode'] = 'same_quality'
        self.controller._h264_requested = True
        with patch.object(self.controller, '_launch_h264') as launch:
            self.controller._sync_bitrate(112, False)
            launch.assert_not_called()
            self.controller.stop()
            self.controller._sync_bitrate(120, True)
            launch.assert_not_called()

    def test_quality_retry_keeps_applied_qp_and_handles_zero(self):
        self.controller.cfg.update(mode='same_quality', h264_qp=40)
        self.controller._h264_requested = True
        self.controller._h264_qp = 0
        self.controller._h264_last_adjustment = 100
        with patch.object(self.controller, '_launch_h264') as launch:
            self.controller._sync_bitrate(112, True)
            launch.assert_not_called()
            self.controller._sync_bitrate(131, True)
            launch.assert_called_once_with(qp=0)


if __name__ == '__main__':
    unittest.main()
