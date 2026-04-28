from __future__ import annotations

import json
from math import ceil
from pathlib import Path

import pandas as pd


CPU_CORES_PER_NODE = 128
GPU_CARD_CAPACITY = 0.80


def _load_phase_payload(phase_dir: Path) -> dict:
    payload_path = phase_dir / "payload.json"
    with payload_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_phase_03_payload(phase_03_dir: Path) -> dict:
    candidate_dirs = [phase_03_dir]
    candidate_dirs.extend(sorted(phase_03_dir.parent.parent.glob("*/phase_03_state_construction")))
    for candidate_dir in candidate_dirs:
        payload_path = candidate_dir / "payload.json"
        if not payload_path.exists():
            continue
        with payload_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("state_cache_path"):
            return payload
    raise FileNotFoundError("Unable to locate a valid phase_03_state_construction payload")


def _gpu_reclaimable_cards(allocated_gpus: float, gpu_util: float) -> int:
    if pd.isna(allocated_gpus) or pd.isna(gpu_util):
        return 0
    allocated = int(allocated_gpus)
    if allocated <= 1:
        return 0

    # Treat gpu_util as per-card average utilization for the allocated cards.
    # Total load is pooled and re-packed so that no remaining card exceeds 80%.
    total_gpu_load = allocated * float(gpu_util)
    cards_needed = max(1, ceil(total_gpu_load / GPU_CARD_CAPACITY))
    cards_needed = min(cards_needed, allocated)
    reclaimable = allocated - cards_needed
    return max(reclaimable, 0)


def run_phase_05(phase_04_dir: Path, scenario_dir: Path, output_dir: Path) -> dict[str, object]:
    phase_04_payload = _load_phase_payload(phase_04_dir)
    phase_03_dir = scenario_dir / "phase_03_state_construction"
    phase_03_payload = _resolve_phase_03_payload(phase_03_dir)

    detections_path = Path(phase_04_payload["underutilized_nodes_path"])
    state_cache_path = Path(phase_03_payload["state_cache_path"])

    detections = pd.read_parquet(
        detections_path,
        columns=[
            "timestamp",
            "node_id",
            "threshold",
            "is_underutilized",
            "cpu_underutilized",
            "gpu_underutilized",
            "cpu_util",
            "gpu_util",
            "memory_util",
        ],
    )
    detections["timestamp"] = pd.to_datetime(detections["timestamp"])
    detections = detections.loc[detections["is_underutilized"]].copy()

    state = pd.read_parquet(
        state_cache_path,
        columns=[
            "timestamp",
            "node_id",
            "cpu_util",
            "gpu_util",
            "memory_util",
            "allocated_cores",
            "allocated_gpus",
            "job_id",
            "active_job_count",
        ],
    )
    state["timestamp"] = pd.to_datetime(state["timestamp"])

    reclamation = detections.merge(
        state,
        on=["timestamp", "node_id", "cpu_util", "memory_util"],
        how="left",
        suffixes=("", "_state"),
    )
    reclamation["reclaimable_cores"] = (
        CPU_CORES_PER_NODE * (1.0 - reclamation["cpu_util"])
    ).where(reclamation["cpu_underutilized"], 0.0).fillna(0.0).astype(int)
    reclamation["reclaimable_cores"] = reclamation["reclaimable_cores"].clip(lower=0)
    reclamation["reclaimable_gpus"] = [
        _gpu_reclaimable_cards(allocated_gpus, gpu_util)
        for allocated_gpus, gpu_util in zip(
            reclamation["allocated_gpus"], reclamation["gpu_util"]
        )
    ]
    reclamation.loc[~reclamation["gpu_underutilized"], "reclaimable_gpus"] = 0
    reclamation["gpu_total_load_estimate"] = (
        reclamation["allocated_gpus"].fillna(0.0) * reclamation["gpu_util"].fillna(0.0)
    )
    reclamation["gpu_cards_needed_after_compaction"] = (
        reclamation["allocated_gpus"].fillna(0.0).astype(int) - reclamation["reclaimable_gpus"]
    ).clip(lower=0)

    reclamation = reclamation[
        [
            "timestamp",
            "node_id",
            "threshold",
            "cpu_util",
            "gpu_util",
            "memory_util",
            "cpu_underutilized",
            "gpu_underutilized",
            "allocated_cores",
            "allocated_gpus",
            "job_id",
            "active_job_count",
            "reclaimable_cores",
            "reclaimable_gpus",
            "gpu_total_load_estimate",
            "gpu_cards_needed_after_compaction",
        ]
    ].sort_values(["timestamp", "node_id"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "reclamation_plan.parquet"
    reclamation.to_parquet(output_path, index=False)

    summary = {
        "status": "complete",
        "reclamation_plan_path": str(output_path),
        "rows_written": int(len(reclamation)),
        "coverage_start": None if reclamation.empty else reclamation["timestamp"].min().isoformat(),
        "coverage_end": None if reclamation.empty else reclamation["timestamp"].max().isoformat(),
        "node_count": int(reclamation["node_id"].nunique()),
        "timestamp_count": int(reclamation["timestamp"].nunique()),
        "rows_with_cpu_reclamation": int((reclamation["reclaimable_cores"] > 0).sum()),
        "rows_with_gpu_reclamation": int((reclamation["reclaimable_gpus"] > 0).sum()),
        "max_reclaimable_cores": int(reclamation["reclaimable_cores"].max()) if not reclamation.empty else 0,
        "max_reclaimable_gpus": int(reclamation["reclaimable_gpus"].max()) if not reclamation.empty else 0,
        "columns": reclamation.columns.tolist(),
        "gpu_reclamation_rule": (
            "total allocated GPU load is estimated as allocated_gpus * gpu_util and compacted "
            "onto the minimum number of cards such that no card exceeds 80% utilization"
        ),
    }
    summary_path = output_dir / "reclamation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary
