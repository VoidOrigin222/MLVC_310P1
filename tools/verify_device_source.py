#!/usr/bin/env python3
"""Verify deployed sources against the inventory generated on the workstation."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
records = json.loads((root / "DEPLOYMENT_SOURCE.json").read_text())["files"]
for record in records:
    data = (root / record["file"]).read_bytes()
    if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise RuntimeError("deployed file mismatch: " + record["file"])
print("Verified {} deployed source/asset files".format(len(records)))
