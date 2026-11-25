#!/usr/bin/env python3
"""Finite encoder-board validation with a local RTP sink and temperature monitoring."""
import argparse
import json
from pathlib import Path
import re
import socket
import subprocess
import threading
import time

from run_dual_board_validation import REPO, ROOT, start, stop, temperature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--sink-port', type=int, default=39346)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9_-]+', args.name) or not 1 <= args.timeout <= 180:
        parser.error('invalid run name or timeout')
    output = REPO / 'artifacts/encoder-only' / args.name
    if output.exists():
        parser.error('run output already exists')
    output.mkdir(parents=True)
    sink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sink.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sink.bind(('0.0.0.0', args.sink_port))
    sink.settimeout(0.2)
    done = threading.Event()
    counters = {'datagrams': 0, 'bytes': 0}

    def receive():
        while not done.is_set():
            try:
                data, address = sink.recvfrom(65535)
            except socket.timeout:
                continue
            if address[0] == '192.168.5.13':
                counters['datagrams'] += 1
                counters['bytes'] += len(data)

    receiver = threading.Thread(target=receive)
    receiver.start()
    process = None
    samples = []
    failure = None
    started = time.monotonic()
    try:
        initial = temperature('310PEncoder')
        if initial['health'] != 'OK' or initial['temperature_c'] >= 85:
            raise RuntimeError('encoder health/temperature limit before start')
        process = start('310PEncoder', 'encode', args.config, args.name)
        last_sample = -5
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > args.timeout:
                raise RuntimeError('encoder test timeout')
            if elapsed - last_sample >= 5:
                sample = {'elapsed_s': elapsed, **temperature('310PEncoder')}
                samples.append(sample)
                print(json.dumps(sample), flush=True)
                last_sample = elapsed
                if sample['health'] != 'OK' or sample['temperature_c'] >= 85:
                    raise RuntimeError('encoder health/temperature limit')
            time.sleep(0.2)
    except BaseException as error:
        failure = str(error) or type(error).__name__
    finally:
        if process is not None:
            if process.poll() is None:
                stop('310PEncoder', 'encode', args.name)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
        done.set()
        receiver.join(timeout=2)
        sink.close()
    metrics = {}
    if process is not None:
        subprocess.run(['scp', f'310PEncoder:{ROOT}/logs/{args.name}-encode.log',
                        str(output / 'encode.log')], check=True, capture_output=True)
        log = (output / 'encode.log').read_text(errors='replace')
        for key, value in re.findall(r'^([a-z_][a-z0-9_]*)=([0-9]+(?:\.[0-9]+)?)$', log, re.M):
            metrics[key] = float(value) if '.' in value else int(value)
        if not failure and process.returncode:
            failure = 'encoder process failed'
        if not failure and (not metrics.get('frames') or metrics.get('detection_frames') != metrics['frames']
                            or 'detection_frame_sync=ok' not in log):
            failure = 'encoded/detected frame association mismatch'
        if not failure and not counters['datagrams']:
            failure = 'local RTP sink received no packets'
    report = {'failure': failure, 'encoder_returncode': process.returncode if process else None,
              'elapsed_s': time.monotonic() - started, 'metrics': metrics,
              'temperature_samples': samples, 'rtp_sink': counters,
              'decoder_started': False, 'boundary': 'encoder only; no RTP reassembly or decoder validation'}
    (output / 'run.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if failure:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
