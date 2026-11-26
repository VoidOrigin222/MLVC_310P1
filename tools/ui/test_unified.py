"""Unification regressions: parameter migration and both real telemetry dialects."""
import json
from pathlib import Path
import socket
import tempfile
import time
import unittest
from unittest.mock import patch
from server import ConfigStore, Counters, Monitor, ROOT, validate_config
from h264 import command


class UnifiedTests(unittest.TestCase):
    def test_both_profiles_have_five_channels(self):
        for name in ('310p', 'thor'):
            cfg = validate_config(json.loads((ROOT / f'ui_config.{name}.json').read_text(encoding='utf-8')))
            self.assertEqual(len(cfg['channels']), 5)
            self.assertEqual(len({c['stats_port'] for c in cfg['channels']}), 5)

    def test_single_channel_config_migrates_preserves_addresses_and_keeps_backup(self):
        original = {'original': 'rtsp://127.0.0.1:8554/custom-camera', 'h264': 'rtsp://127.0.0.1:8554/custom-h264',
                    'semantic': 'rtsp://127.0.0.1:8554/custom-ulbvc', 'webrtc': 'http://127.0.0.1:8889',
                    'transport': 'udp', 'h264_qp': 37, 'ffmpeg': 'auto', 'stats_port': 39601}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ui_config.json'
            path.write_text(json.dumps(original), encoding='utf-8')
            previous = path.read_bytes()
            cfg = ConfigStore(path).snapshot()
            self.assertEqual(len(cfg['channels']), 5)
            for key in ('original', 'h264', 'semantic'):
                self.assertEqual(cfg['channels'][0][key], original[key])
            self.assertEqual(cfg['h264_transport'], 'udp')
            self.assertEqual(cfg['h264_qp'], 37)
            self.assertEqual(cfg['channels'][0]['stats_port'], 39601)
            self.assertEqual(path.with_name(path.name + '.v0.5-backup').read_bytes(), previous)
            self.assertEqual(ConfigStore(path).snapshot(), cfg)

    def test_payload_and_wire_rates_remain_separate(self):
        cfg = json.loads((ROOT / 'ui_config.310p.json').read_text(encoding='utf-8'))
        counters = Counters(); ch = cfg['channels'][0]
        for n, now in ((1, 10), (2, 11)):
            self.assertTrue(counters.record('ch1', {'kind': 'ulbvc_rtp_stats', 'frames': 30*n,
                'payload_bytes': 1000*n, 'wire_bytes': 1800*n}, now))
            self.assertTrue(counters.record('ch2', {'kind': 'semantic_codec_stats', 'frames': 12*n,
                'payload_bytes': 2000*n}, now))
        self.assertEqual(counters.metrics(ch, 11)['kbs'], 1)
        self.assertEqual(counters.metrics(ch, 11)['wire_kbs'], 1.8)
        self.assertEqual(counters.metrics(cfg['channels'][1], 11)['kbs'], 2)
        self.assertIsNone(counters.metrics(cfg['channels'][1], 11)['wire_kbs'])
        self.assertIsNone(counters.metrics(ch, 17))
        counters.record('ch1', {'kind': 'semantic_codec_stats', 'frames': 70, 'payload_bytes': 4000}, 18)
        self.assertIsNone(counters.metrics(ch, 18))
        counters.clear(); self.assertIsNone(counters.metrics(cfg['channels'][1], 18))

    def test_software_comparison_preserves_qp_and_transport(self):
        cfg = json.loads((ROOT / 'ui_config.310p.json').read_text(encoding='utf-8'))
        cfg['h264_transport'] = 'udp'
        args = command(cfg, cfg['channels'][0], 'libx264')
        self.assertEqual(args[args.index('-qp') + 1], '40')
        self.assertNotIn('-hwaccel', args)
        self.assertEqual(args.count('udp'), 2)
        self.assertNotIn('-b:v', args)

    def test_udp_receives_both_devices_on_separate_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = json.loads((ROOT / 'ui_config.310p.json').read_text(encoding='utf-8'))
            holders = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in cfg['channels']]
            for ch, sock in zip(cfg['channels'], holders):
                sock.bind(('127.0.0.1', 0)); ch['stats_port'] = sock.getsockname()[1]
            for sock in holders: sock.close()
            store = ConfigStore(Path(directory) / 'ui_config.json'); store.save(cfg)
            with patch.object(Monitor, 'maintain_h264', return_value=None): monitor = Monitor(store)
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                deadline = time.monotonic() + 3
                while len(monitor.sockets) != 5 and time.monotonic() < deadline: time.sleep(.02)
                self.assertEqual(len(monitor.sockets), 5)
                for n in range(1, 4):
                    for index, kind in enumerate(('ulbvc_rtp_stats', 'semantic_codec_stats')):
                        sender.sendto(json.dumps({'kind': kind, 'frames': n*30, 'payload_bytes': n*1000}).encode(),
                                      ('127.0.0.1', cfg['channels'][index]['stats_port']))
                    time.sleep(.08)
                deadline = time.monotonic() + 2
                while monitor.counters.metrics(cfg['channels'][1]) is None and time.monotonic() < deadline: time.sleep(.02)
                for ch in cfg['channels'][:2]: self.assertIsNotNone(monitor.counters.metrics(ch))
                self.assertIsNone(monitor.counters.metrics(cfg['channels'][2]))
            finally:
                sender.close(); monitor.close()


if __name__ == '__main__': unittest.main()
