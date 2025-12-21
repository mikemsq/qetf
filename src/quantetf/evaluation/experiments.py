from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import hashlib
import json


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


@dataclass(frozen=True)
class ExperimentManifest:
    run_id: str
    dataset_id: str
    config_hashes: Mapping[str, str]
    notes: str = ""


def write_manifest(path: Path, manifest: ExperimentManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.__dict__, f, indent=2)
