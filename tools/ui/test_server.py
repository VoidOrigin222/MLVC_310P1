"""Configuration safety, per-channel counters and WHEP ownership regression."""

import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.parse import urlsplit

from server import (
    ConfigStore,
    Counters,
    Monitor,
    ROOT,
    make_handler,
    validate_config,
)


def defaults():
    return validate_config(json.loads((ROOT / "ui_config.json").read_text(encoding="utf8")))


class ConfigTests(unittest.TestCase):
    def test_variable_channel_counts_and_mixed_fps_survive_restart(self):
        for rates in ([30], [30, 30], [20] * 3, [30, 30, 20, 20, 20], [12] * 6):
            with self.subTest(rates=rates), tempfile.TemporaryDirectory() as directory:
                cfg = defaults()
                template = cfg["channels"][0]
                cfg["channels"] = []
                for index, fps in enumerate(rates):
                    channel = copy.deepcopy(template)
                    channel.update(id=f"ch{index + 1}", stats_port=39600 + index, fps=fps)
                    cfg["channels"].append(channel)
                store = ConfigStore(Path(directory) / "parameters.json")
                store.save(cfg)
                self.assertEqual(ConfigStore(store.path).snapshot(), cfg)
        for channels in ([], None, 2):
            cfg = defaults()
            cfg["channels"] = channels
            with self.assertRaises(ValueError):
                validate_config(cfg)

    def test_restart_reads_file_instead_of_browser_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parameters.json"
            store = ConfigStore(path)
            cfg = store.snapshot()
            cfg["channels"][4][
                "semantic"
            ] = "rtsp://192.168.5.3:8554/custom/fifth"
            store.save(cfg)
            self.assertEqual(ConfigStore(path).snapshot(), cfg)

    def test_failed_atomic_save_keeps_previous_file_and_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ConfigStore(Path(directory) / "parameters.json")
            previous = store.snapshot()
            saved = store.path.read_bytes()
            cfg = copy.deepcopy(previous)
            cfg["channels"][0]["name"] = "测试通道"
            with patch(
                "server.os.replace", side_effect=PermissionError("locked")
            ):
                with self.assertRaises(PermissionError):
                    store.save(cfg)
            self.assertEqual(store.snapshot(), previous)
            self.assertEqual(store.path.read_bytes(), saved)

    def test_rejects_ambiguous_ports_ids_and_invalid_urls(self):
        for field, value in (
            ("stats_port", 39601),
            ("id", "ch2"),
            ("semantic", "rtsp://127.0.0.1:8554"),
            ("original", "file:///private"),
        ):
            cfg = defaults()
            cfg["channels"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_config(cfg)


class CounterTests(unittest.TestCase):
    def packet(self, frames, count):
        return {
            "kind": "semantic_codec_stats",
            "frames": frames,
            "payload_bytes": count,
        }

    def test_real_counter_rate_and_channel_isolation(self):
        counters = Counters()
        channel = defaults()["channels"][0]
        counters.record("ch1", self.packet(1, 1000), 10)
        counters.record("ch1", self.packet(13, 4000), 11)
        value = counters.metrics(channel, 11)
        self.assertAlmostEqual(value["kbs"], 3)
        self.assertAlmostEqual(value["encode_fps"], 12)
        self.assertAlmostEqual(value["bpp"], 3000 * 8 / (12 * 1920 * 1080))
        self.assertIsNone(counters.metrics(defaults()["channels"][1], 11))

    def test_encoder_restart_and_stale_values_clear_metrics(self):
        counters = Counters()
        channel = defaults()["channels"][0]
        counters.record("ch1", self.packet(120, 100000), 10)
        counters.record("ch1", self.packet(132, 110000), 11)
        self.assertIsNone(counters.metrics(channel, 17))
        counters.record("ch1", self.packet(1, 1000), 18)
        self.assertIsNone(counters.metrics(channel, 18))
        counters.record("ch1", self.packet(13, 4000), 19)
        self.assertAlmostEqual(counters.metrics(channel, 19)["kbs"], 3)

    def test_rejects_invalid_datagrams(self):
        counters = Counters()
        for value in (
            None,
            {"kind": "other"},
            self.packet(True, 1),
            self.packet(1, -1),
            self.packet(1, 10**400),
        ):
            self.assertFalse(counters.record("ch1", value, 10))


class HttpTests(unittest.TestCase):
    def test_whep_proxy_owns_and_releases_only_created_sessions(self):
        requests = []

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                requests.append(
                    (
                        "POST",
                        self.path,
                        self.rfile.read(int(self.headers["Content-Length"])),
                    )
                )
                self.send_response(201)
                self.send_header("Location", self.path + "/owned-session")
                self.end_headers()
                self.wfile.write(b"v=0\r\n")

            def do_DELETE(self):
                requests.append(("DELETE", self.path))
                self.send_response(200)
                self.end_headers()

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        worker = threading.Thread(target=upstream.serve_forever, daemon=True)
        worker.start()
        with tempfile.TemporaryDirectory() as directory:
            store = ConfigStore(Path(directory) / "parameters.json")
            cfg = store.snapshot()
            cfg["webrtc"] = f"http://127.0.0.1:{upstream.server_port}"
            store.save(cfg)
            # Tests do not acquire the real desktop UI's telemetry ports.
            with patch.object(Monitor, "receive_stats", return_value=None):
                monitor = Monitor(store)
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), make_handler(monitor)
            )
            server.daemon_threads = True
            ui_worker = threading.Thread(
                target=server.serve_forever, daemon=True
            )
            ui_worker.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(
                    Request(
                        base + "/api/whep/ch1/semantic",
                        data=b"offer",
                        headers={"Content-Type": "application/sdp"},
                    )
                ) as response:
                    self.assertEqual(response.status, 201)
                    location = response.headers["Location"]
                    self.assertTrue(location.startswith("/api/session/"))
                self.assertEqual(len(monitor.sessions), 1)
                with urlopen(Request(base + location, method="DELETE")):
                    pass
                self.assertEqual(len(monitor.sessions), 0)
                self.assertEqual(
                    requests[0], ("POST", urlsplit(cfg["channels"][0]["semantic"]).path + "/whep", b"offer")
                )
                self.assertEqual(
                    requests[1], ("DELETE", urlsplit(cfg["channels"][0]["semantic"]).path + "/whep/owned-session")
                )
                with urlopen(base + "/api/health") as response:
                    self.assertEqual(
                        json.load(response)["app"], "ULBVCMonitor"
                    )
            finally:
                server.shutdown()
                server.server_close()
                monitor.close()
                ui_worker.join(2)
        upstream.shutdown()
        upstream.server_close()
        worker.join(2)


if __name__ == "__main__":
    unittest.main()
