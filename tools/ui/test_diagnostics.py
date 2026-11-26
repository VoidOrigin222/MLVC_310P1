"""Remote-machine diagnosis without UI sockets, streaming or config writes."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from diagnostics import diagnose


class DiagnosticsTests(unittest.TestCase):
    def test_reports_real_failures_without_changing_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "parameters.json"
            config.write_text(json.dumps({"h264_encoder": "auto"}), encoding="utf8")
            previous = config.read_bytes()
            report = Path(directory) / "diagnostics.json"
            with patch("diagnostics.video_controllers", return_value=[{"Name": "AMD Radeon"}]), patch(
                "diagnostics.probe_encoder",
                side_effect=[
                    {"available": False, "detail": "No Intel device"},
                    {"available": False, "detail": "No NVIDIA device"},
                    {"available": False, "detail": "AMF runtime missing"},
                ],
            ):
                self.assertEqual(diagnose(config, report), 0)
            value = json.loads(report.read_text(encoding="utf8"))
            self.assertFalse(value["hardware_encoder_available"])
            self.assertIn("AMF runtime missing", value["encoders"]["h264_amf"]["detail"])
            self.assertEqual(config.read_bytes(), previous)

    def test_invalid_parameters_still_produce_a_diagnostic_report(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "broken.json"
            config.write_text("{", encoding="utf8")
            report = Path(directory) / "diagnostics.json"
            with patch("diagnostics.video_controllers", return_value=[]):
                self.assertEqual(diagnose(config, report), 1)
            self.assertIn("error", json.loads(report.read_text(encoding="utf8")))


if __name__ == "__main__":
    unittest.main()
