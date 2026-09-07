#!/usr/bin/env python3
"""Regression checks for the ACL stage-benchmark acceptance report."""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_file(path: Path, payload: bytes = b"x") -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"bytes": len(payload), "sha256": sha256_bytes(payload)}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_source_manifest(source_manifest: Path) -> None:
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        source_manifest,
        {
            "created_at": "2026-05-16T00:00:00Z",
            "dtype": "fp16",
            "sidecar": {"file": "sidecars.mlvcsc", "bytes": 1, "sha256": sha256_bytes(b"s")},
            "models": [{"name": "stage_a", "file": "stage_a.sim.onnx"}],
        },
    )


def make_acl_manifest(root: Path) -> Path:
    sidecar = write_file(root / "sidecars.mlvcsc", b"s")
    write_file(root / "onnx_original" / "stage_a.sim.onnx", b"o")
    write_file(root / "onnx_optimized" / "stage_a.sim.onnx", b"p")
    write_file(root / "om_atc" / "stage_a.sim.om", b"a")
    aoe = write_file(root / "om_aoe" / "stage_a.sim.om", b"e")
    write_file(root / "reports" / "io" / "stage_a.sim.atc.json", b"{}")
    write_file(root / "reports" / "io" / "stage_a.sim.aoe.json", b"{}")
    write_json(root / "reports" / "rewrite" / "stage_a.sim.json", {"match_count": 0})
    manifest = {
        "created_at": "2026-05-16T00:00:00Z",
        "runtime": "acl",
        "soc_version": "Ascend310P3",
        "dtype": "fp16",
        "sidecar": {"file": "sidecars.mlvcsc", **sidecar},
        "models": [
            {
                "name": "stage_a",
                "file": "om_aoe/stage_a.sim.om",
                "onnx_file": "onnx_original/stage_a.sim.onnx",
                "optimized_onnx_file": "onnx_optimized/stage_a.sim.onnx",
                "atc_om_file": "om_atc/stage_a.sim.om",
                "backend": "acl",
                "bytes": aoe["bytes"],
                "sha256": aoe["sha256"],
                "frame_type": "I",
                "route": "encode",
                "optimizations": {
                    "post_sigmoid_chunk_add": 0,
                    "fp16_io": True,
                    "aoe": "highest",
                    "aoe_tune_optimization_level": "O1",
                },
                "inputs": [{"name": "x", "dtype": "float16", "shape": [1, 1, 1, 1]}],
                "outputs": [{"name": "y", "dtype": "float16", "shape": [1, 1, 1, 1]}],
            }
        ],
    }
    write_json(root / "manifest.json", manifest)
    return root / "manifest.json"


def make_fake_benchmark(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "stage = sys.argv[sys.argv.index('--stage') + 1]\n"
        "print(f'stage={stage}')\n"
        "print('warmup=50')\n"
        "print('iterations=2000')\n"
        "print('total_ms=20')\n"
        "print('avg_ms=0.01')\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mlvc_acl_stage_benchmark_") as temp:
        root = Path(temp)
        source_manifest = root / "source" / "720p" / "manifest.json"
        acl_manifest = make_acl_manifest(root / "models" / "720p_acl")
        make_source_manifest(source_manifest)
        fake_benchmark = root / "fake_benchmark.py"
        make_fake_benchmark(fake_benchmark)
        output = root / "acceptance" / "stage_performance.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "run_acl_stage_benchmark.py"),
            "--root",
            str(root),
            "--resolution",
            "720p",
            "--manifest",
            str(acl_manifest),
            "--source-manifest",
            str(source_manifest),
            "--benchmark-binary",
            str(fake_benchmark),
            "--output",
            str(output),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        assert_true(completed.returncode == 0, completed.stderr or completed.stdout)
        report = json.loads(output.read_text(encoding="utf-8"))
        assert_true(report["status"] == "passed", "report should pass")
        assert_true(report["gate"] == "stage_performance", "bad gate")
        assert_true(report["benchmarked_stage_count"] == 1, "stage was not benchmarked")
        assert_true(report["metrics"]["avg_ms_max"] == 0.01, "bad avg_ms")

    print("run_acl_stage_benchmark_check status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
