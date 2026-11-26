#!/usr/bin/env python3
"""Build/run CPU-only protocol, socket and rate tests using a POSIX C++ compiler."""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    'common/src/core/status.cc', 'common/src/transport/rtcp_session.cc',
    'common/src/transport/rate_feedback.cc', 'common/src/transport/network_rate_controller.cc',
    'common/src/transport/rtp_ulbvc.cc', 'common/src/transport/rtp_message_receiver.cc',
    'common/src/transport/ulbvc_media_unit.cc', 'common/src/transport/udp_pacer.cc',
    'common/src/transport/detection_metadata.cc',
    'ulbvc/src/codec/rate_control/ulbvc_rate_control.cc',
]
TESTS = ['test_rtcp_session', 'test_network_rate_control', 'test_runtime_rate_control',
         'test_rtp_ulbvc', 'test_rtp_reassembler', 'test_rtp_message_receiver_sequence',
         'test_rtp_rtcp_session', 'test_adaptive_rtcp_session', 'test_udp_pacer_rate',
         'test_detection_metadata']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', default='g++')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/rtcp-local/tests')
    parser.add_argument('--openssl', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    compiler = str(Path(args.compiler).resolve()) if Path(args.compiler).is_file() else args.compiler
    env = os.environ.copy()
    env['PATH'] = str(Path(compiler).parent) + os.pathsep + env.get('PATH', '')
    env.pop('ULBVC_RTCP_KEY', None)
    flags = ['-std=gnu++17', '-O2', '-g', '-pthread', '-Wall', '-Wextra',
             '-Icommon/include', '-Iulbvc/include']
    if args.openssl:
        flags += ['-DULBVC_HAS_OPENSSL=1']
    records = []
    (output / 'results.json').write_text('[]\n', encoding='utf-8')
    with (output / 'build.log').open('wb') as log:
        objects = []
        for source in SOURCES:
            obj = output / (Path(source).stem + '.o')
            command = [compiler, *flags, '-c', source, '-o', obj.as_posix()]
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=120, check=True)
            objects.append(obj.as_posix())
        for name in TESTS + (['test_rtcp_authentication'] if args.openssl else []):
            binary = output / (name + ('.exe' if os.name == 'nt' else ''))
            subprocess.run([compiler, *flags, f'tools/cpp/tests/{name}.cc', *objects,
                            *(['-lcrypto'] if args.openssl else []), '-o', binary.as_posix()],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=120, check=True)
            started = time.monotonic()
            with (output / f'{name}.log').open('wb') as test_log:
                test_env = env.copy()
                if args.openssl and name == 'test_adaptive_rtcp_session':
                    test_env['ULBVC_RTCP_KEY'] = secrets.token_hex(32)
                result = subprocess.run([str(binary)], cwd=ROOT, env=test_env, stdout=test_log,
                                        stderr=subprocess.STDOUT, timeout=45)
            record = {'test': name, 'returncode': result.returncode,
                      'seconds': time.monotonic() - started}
            records.append(record)
            (output / 'results.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
            print(json.dumps(record), flush=True)
            if result.returncode:
                raise SystemExit(f'{name} failed; see {output / (name + ".log")}')


if __name__ == '__main__':
    main()
