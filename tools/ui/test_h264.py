"""Hardware baseline settings, legacy migration and owned-process lifecycle."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from h264 import Publishers, command, encoder_options, probe_encoder
from server import ConfigStore, ROOT, validate_config


def defaults():
    return json.loads((ROOT / "ui_config.thor.json").read_text(encoding="utf8"))


class H264Tests(unittest.TestCase):
    def test_rtsp_clock_is_not_used_for_cfr_duplication(self):
        cfg = defaults()
        for encoder in ("libx264", "h264_qsv", "h264_nvenc", "h264_amf"):
            args = command(cfg, cfg["channels"][0], encoder)
            self.assertEqual(args[args.index("-fps_mode:v") + 1], "passthrough")
            self.assertEqual(args[args.index("-enc_time_base:v") + 1], "demux")
            self.assertGreater(args.index("-fps_mode:v"), args.index("-i"))
            self.assertEqual(args[args.index("-fflags") + 1], "nobuffer")
            self.assertNotIn("-r", args)
            self.assertNotIn("-fpsmax:v", args)  # Conflicts with passthrough in FFmpeg 8.

    def test_lost_input_stops_only_owned_job_but_api_outage_keeps_it(self):
        cfg = defaults()
        manager = Publishers()
        manager.encoder = "h264_qsv"
        process = Mock()
        process.poll.return_value = None
        try:
            with patch.object(manager, "probe"), patch("h264.subprocess.Popen", return_value=process):
                manager.update(cfg, {"input0"})
                manager.update(cfg, None)
                self.assertIn("ch1", manager.jobs)
                process.terminate.assert_not_called()
                manager.update(cfg, set())
                self.assertNotIn("ch1", manager.jobs)
                process.terminate.assert_called_once()
        finally:
            manager.close()

    def test_legacy_file_migrates_without_changing_existing_addresses(self):
        cfg = defaults()
        for key in list(cfg):
            if key.startswith("h264_"):
                del cfg[key]
        for c in cfg["channels"]:
            del c["h264"]
        cfg["channels"][0]["original"] = "rtsp://192.168.5.3:8554/custom/source"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parameters.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            current = ConfigStore(path).snapshot()
            self.assertEqual(current["h264_qp"], 40)
            self.assertEqual(
                current["channels"][0]["original"], cfg["channels"][0]["original"]
            )
            self.assertEqual(
                current["channels"][0]["h264"], "rtsp://192.168.5.3:8554/camera-h2640"
            )
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), current)

    def test_invalid_qp_and_overwriting_source_are_rejected(self):
        for value in (-1, 52, True, "40"):
            cfg = defaults()
            cfg["h264_qp"] = value
            with self.assertRaises(ValueError):
                validate_config(cfg)
        cfg = defaults()
        cfg["channels"][0]["h264"] = cfg["channels"][0]["original"]
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_commands_use_hardware_and_constant_qp(self):
        cfg = defaults()
        cfg["h264_qp"] = 35
        for encoder, flag in (
            ("h264_qsv", "-q:v"),
            ("h264_nvenc", "-qp"),
            ("h264_amf", "-qp_p"),
        ):
            args = command(cfg, cfg["channels"][0], encoder)
            self.assertEqual(args[args.index(flag) + 1], "35")
            self.assertIn("-hwaccel", args)
            thread_index = args.index("-threads:v")
            self.assertEqual(args[thread_index + 1], "1")
            self.assertLess(thread_index, args.index("-i"))
            self.assertIn("-xerror", args)
            self.assertNotIn("libx264", args)
            self.assertNotIn("-b:v", args)
            self.assertNotEqual(args[args.index("-i") + 1], args[-1])
            if encoder == "h264_qsv":
                self.assertIn("+qscale", args)

    def test_amf_config_survives_save_and_restart(self):
        cfg = defaults()
        cfg["h264_encoder"] = "h264_amf"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parameters.json"
            ConfigStore(path).save(cfg)
            self.assertEqual(ConfigStore(path).snapshot(), cfg)

    def test_amf_keeps_gpu_surfaces_and_limits_encoder_queue(self):
        cfg = defaults()
        cfg["h264_qp"] = 28
        args = command(cfg, cfg["channels"][0], "h264_amf")
        for flag, value in (
            ("-hwaccel", "d3d11va"),
            ("-hwaccel_output_format", "d3d11"),
            ("-usage", "ultralowlatency"),
            ("-rc", "cqp"),
            ("-qp_i", "28"),
            ("-qp_p", "28"),
            ("-qp_b", "28"),
            ("-async_depth", "1"),
            ("-preanalysis", "0"),
            ("-preencode", "0"),
            ("-frame_skipping", "0"),
            ("-bf", "0"),
        ):
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertLess(args.index("-hwaccel_output_format"), args.index("-i"))
        for unwanted in ("cuda", "hwdownload", "libx264", "-rc-lookahead"):
            self.assertNotIn(unwanted, args)

    def test_auto_probes_amf_after_other_hardware_fails(self):
        cfg = defaults()
        manager = Publishers()
        with patch("h264.os.name", "nt"), patch(
            "h264.subprocess.run",
            side_effect=[
                Mock(returncode=1, stderr=b"No Intel device"),
                Mock(returncode=1, stderr=b"No NVIDIA device"),
                Mock(returncode=0, stderr=b""),
            ],
        ) as run:
            manager.probe(cfg)
            self.assertEqual(manager.encoder, "h264_amf")
            self.assertEqual(manager.error, "")
            self.assertEqual(
                [call.args[0][call.args[0].index("-c:v") + 1] for call in run.call_args_list],
                ["h264_qsv", "h264_nvenc", "h264_amf"],
            )
            args = run.call_args.args[0]
            options = encoder_options(cfg, "h264_amf")
            index = args.index("-c:v")
            self.assertEqual(args[index:index + len(options)], options)
            manager.probe(cfg)
            self.assertEqual(run.call_count, 3)

    def test_explicit_amf_does_not_switch_to_other_encoders(self):
        cfg = defaults()
        cfg["h264_encoder"] = "h264_amf"
        manager = Publishers()
        with patch(
            "h264.subprocess.run", return_value=Mock(returncode=1, stderr=b"AMF device unavailable")
        ) as run:
            manager.probe(cfg)
            self.assertIsNone(manager.encoder)
            self.assertTrue(manager.error)
            self.assertEqual(run.call_count, 1)
            self.assertIn("h264_amf", run.call_args.args[0])
            manager.probe(cfg)
            self.assertEqual(run.call_count, 1)
            self.assertIn("AMF device unavailable", manager.status()["probes"]["h264_amf"]["detail"])

    def test_probe_preserves_driver_failure_and_timeout(self):
        import subprocess

        with patch(
            "h264.subprocess.run",
            return_value=Mock(returncode=1, stderr=b"DLL amfrt64.dll failed to open"),
        ) as run:
            result = probe_encoder(defaults(), "h264_amf")
            self.assertFalse(result["available"])
            self.assertIn("amfrt64.dll", result["detail"])
            self.assertIn("color=c=black:s=640x480:r=30,format=nv12", run.call_args.args[0])
        with patch("h264.subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 8)):
            result = probe_encoder(defaults(), "h264_amf")
            self.assertFalse(result["available"])
            self.assertIsNone(result["returncode"])

    def test_unknown_encoder_cannot_create_a_publisher_command(self):
        cfg = defaults()
        cfg["h264_encoder"] = "invalid_encoder"
        with self.assertRaises(ValueError):
            validate_config(cfg)
        with self.assertRaises(ValueError):
            command(cfg, cfg["channels"][0], "invalid_encoder")

    def test_qp_change_restarts_only_owned_publishers(self):
        cfg = defaults()
        manager = Publishers()
        manager.encoder = "h264_qsv"
        fake1, fake2 = Mock(), Mock()
        fake1.poll.return_value = fake2.poll.return_value = None
        try:
            with patch.object(manager, "probe"), patch(
                "h264.subprocess.Popen", side_effect=[fake1, fake2]
            ) as popen:
                manager.update(cfg, {"input0"})
                manager.update(cfg, {"input0"})
                self.assertEqual(popen.call_count, 1)
                cfg["h264_qp"] = 35
                manager.update(cfg, {"input0"})
                self.assertEqual(popen.call_count, 2)
                fake1.terminate.assert_called_once()
                manager.close()
                fake2.terminate.assert_called_once()
        finally:
            manager.close()

    def test_external_stream_is_not_replaced_and_closed_manager_stays_closed(self):
        cfg = defaults()
        manager = Publishers()
        manager.encoder = "h264_qsv"
        with patch.object(manager, "probe"), patch("h264.subprocess.Popen") as popen:
            manager.update(cfg, {"input0", "camera-h2640"})
            popen.assert_not_called()
            manager.close()
            manager.update(cfg, {"input0"})
            popen.assert_not_called()

    def test_removed_channel_stops_even_when_remaining_sources_are_offline(self):
        cfg = defaults()
        manager = Publishers()
        manager.encoder = "h264_qsv"
        process = Mock()
        process.poll.return_value = None
        try:
            with patch.object(manager, "probe"), patch(
                "h264.subprocess.Popen", return_value=process
            ):
                manager.update(cfg, {"input4"})
                self.assertIn("ch5", manager.jobs)
                cfg["channels"] = cfg["channels"][:2]
                manager.update(cfg, set())
                self.assertNotIn("ch5", manager.jobs)
                process.terminate.assert_called_once()
        finally:
            manager.close()


if __name__ == "__main__":
    unittest.main()
