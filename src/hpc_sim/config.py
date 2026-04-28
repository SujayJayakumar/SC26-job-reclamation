from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    root_dir = path.parent.parent.resolve()
    config["_config_path"] = str(path)
    config["_root_dir"] = str(root_dir)
    return config


def _resolve_case_insensitive(base: Path, relative_path: str) -> Path:
    current = base
    for raw_part in Path(relative_path).parts:
        candidate = current / raw_part
        if candidate.exists():
            current = candidate
            continue

        lowered = raw_part.lower()
        matches = [child for child in current.iterdir() if child.name.lower() == lowered]
        if matches:
            current = matches[0]
            continue

        current = candidate
    return current


def resolve_repo_path(config: dict[str, Any], relative_path: str) -> Path:
    root_dir = Path(config["_root_dir"])
    return _resolve_case_insensitive(root_dir, relative_path)
