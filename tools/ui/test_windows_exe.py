"""Portable executable smoke test using the launcher's isolated self-test."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('executable', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    env = os.environ.copy()
    system = Path(env.get('SystemRoot', 'C:/Windows'))
    env['PATH'] = str(system / 'System32') + os.pathsep + str(system)
    with tempfile.TemporaryDirectory(prefix='ulbvc-ui-exe-') as directory:
        report = Path(directory) / 'result.json'
        run = subprocess.run([str(args.executable.resolve()), '--self-test', str(report)],
                             env=env, timeout=90, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        value = json.loads(report.read_text(encoding='utf-8')) if report.exists() else {'passed': False, 'exit_code': run.returncode}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    if run.returncode or not value.get('passed'):
        raise SystemExit('Executable self-test failed: ' + str(args.report))
    print('Portable executable self-test passed')


if __name__ == '__main__':
    main()
