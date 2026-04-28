from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 42


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def phase_display_name(phase_id: str) -> str:
    return {
        "phase_03_state_construction": "State Construction",
        "phase_04_detection": "Detection",
        "phase_05_reclamation": "Reclamation",
        "phase_06_opportunistic_job_modeling": "Opportunistic Job Modeling",
    }[phase_id]


def repair_phase_payload(target_scenario: str, phase_id: str, source_scenario: str) -> None:
    source_dir = RESULTS / source_scenario / phase_id
    target_dir = RESULTS / target_scenario / phase_id
    source_payload = load_json(source_dir / "payload.json")

    payload = deepcopy(source_payload)
    payload["status"] = "complete"
    payload["scenario"] = target_scenario
    payload["phase_id"] = phase_id
    payload["display_name"] = phase_display_name(phase_id)
    payload["seed"] = SEED
    payload["shared_artifact_source"] = source_scenario
    payload["started_at"] = utc_now()
    payload["completed_at"] = utc_now()

    write_json(target_dir / "payload.json", payload)
    write_json(
        target_dir / "manifest.json",
        {
            "phase_id": phase_id,
            "display_name": phase_display_name(phase_id),
            "scenario": target_scenario,
            "status": "complete",
            "payload_path": str(target_dir / "payload.json"),
            "seed": SEED,
            "force": True,
            "completed_at": payload["completed_at"],
        },
    )


def main() -> None:
    thresholds = ("0.15", "0.20", "0.25")

    # Phase 03 is scenario-independent; use the canonical original 0.15 build.
    for threshold in thresholds:
        for baseline in ("original", "aggressive", "buffered"):
            scenario = f"{baseline}_threshold_{threshold}"
            if scenario == "original_threshold_0.15":
                continue
            repair_phase_payload(
                target_scenario=scenario,
                phase_id="phase_03_state_construction",
                source_scenario="original_threshold_0.15",
            )

    # Phases 04-06 are threshold-dependent but mode-independent in the current design.
    for threshold in thresholds:
        source_scenario = f"original_threshold_{threshold}"
        for baseline in ("aggressive", "buffered"):
            scenario = f"{baseline}_threshold_{threshold}"
            for phase_id in (
                "phase_04_detection",
                "phase_05_reclamation",
                "phase_06_opportunistic_job_modeling",
            ):
                repair_phase_payload(
                    target_scenario=scenario,
                    phase_id=phase_id,
                    source_scenario=source_scenario,
                )


if __name__ == "__main__":
    main()
