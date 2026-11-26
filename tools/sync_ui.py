"""Synchronize the shared UI into another checkout, retaining its local parameters."""
import argparse
import hashlib
from pathlib import Path
import shutil

SOURCE = Path(__file__).resolve().parent / 'ui'
FILES = ('app.js', 'index.html', 'styles.css', 'server.py', 'h264.py', 'diagnostics.py',
         'launch_ui.py', 'web_ui.py', 'build_windows_exe.py', 'requirements-build.txt',
         'run_ui.bat', 'run_web_ui.bat', 'README.md', 'ui_config.310p.json', 'ui_config.thor.json',
         'test_server.py', 'test_h264.py', 'test_unified.py', 'test_diagnostics.py',
         'test_windows_exe.py', 'test_frame_delay.cjs', 'test_player_recovery.cjs', 'test_connection_recovery.cjs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', type=Path, help='Destination tools/ui directory')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(); target = args.target.resolve()
    if target == SOURCE.resolve(): parser.error('Target must be another checkout')
    changed = []
    for name in FILES:
        src, dst = SOURCE / name, target / name
        if dst.is_file() and src.read_bytes() == dst.read_bytes(): continue
        changed.append(name)
        if not args.check:
            target.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    if args.check and changed: raise SystemExit('UI sources differ: ' + ', '.join(changed))
    digest = hashlib.sha256(b''.join((SOURCE / name).read_bytes() for name in FILES)).hexdigest()
    print(f'Shared UI files={len(FILES)} changed={len(changed)} sha256={digest}')


if __name__ == '__main__': main()
