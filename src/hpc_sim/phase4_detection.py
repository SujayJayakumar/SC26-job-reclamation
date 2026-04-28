from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


SCENARIO_THRESHOLD_RE = re.compile(r"_threshold_(\d+\.\d+)$")


def _load_phase_03_payload(phase_03_dir: Path) -> dict:
    payload_path = phase_03_dir / "payload.json"
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


def _threshold_from_scenario(scenario: str) -> float:
    match = SCENARIO_THRESHOLD_RE.search(scenario)
    if not match:
        raise ValueError(f"Unable to parse threshold from scenario: {scenario}")
    return float(match.group(1))


def run_phase_04(input_dir: Path, output_dir: Path, scenario: str) -> dict[str, object]:
    phase_03_payload = _resolve_phase_03_payload(input_dir)
    state_cache_path = Path(phase_03_payload["state_cache_path"])
    threshold = _threshold_from_scenario(scenario)

    state = pd.read_parquet(
        state_cache_path,
        columns=[
            "timestamp",
            "node_id",
            "cpu_util",
            "gpu_util",
            "memory_util",
            "allocated_gpus",
            "telemetry_present_any",
        ],
    )
    state["timestamp"] = pd.to_datetime(state["timestamp"])
    state = state.sort_values(["node_id", "timestamp"]).reset_index(drop=True)

    grouped = state.groupby("node_id", sort=False)
    state["previous_timestamp"] = grouped["timestamp"].shift(1)
    state["previous_cpu_util"] = grouped["cpu_util"].shift(1)
    state["previous_gpu_util"] = grouped["gpu_util"].shift(1)
    state["previous_memory_util"] = grouped["memory_util"].shift(1)
    state["previous_allocated_gpus"] = grouped["allocated_gpus"].shift(1)
    state["previous_telemetry_present_any"] = grouped["telemetry_present_any"].shift(1)

    state["has_previous_interval"] = (
        (state["timestamp"] - state["previous_timestamp"]) == pd.Timedelta(minutes=10)
    )
    state["cpu_condition_current"] = state["cpu_util"] < threshold
    state["cpu_condition_previous"] = state["previous_cpu_util"] < threshold
    state["gpu_condition_current"] = (
        (state["allocated_gpus"].fillna(0.0) > 1.0)
        & (state["gpu_util"] < 0.8)
    )
    state["gpu_condition_previous"] = (
        (state["previous_allocated_gpus"].fillna(0.0) > 1.0)
        & (state["previous_gpu_util"] < 0.8)
    )
    state["memory_condition_current"] = state["memory_util"] < 0.50
    state["memory_condition_previous"] = state["previous_memory_util"] < 0.50
    state["memory_spike_detected"] = state["memory_util"] > (
        2.0 * state["previous_memory_util"]
    )
    state["memory_stable"] = (
        state["telemetry_present_any"]
        & state["previous_telemetry_present_any"].fillna(False)
        & state["has_previous_interval"]
        & state["memory_condition_current"]
        & state["memory_condition_previous"].fillna(False)
        & ~state["memory_spike_detected"].fillna(False)
    )
    state["cpu_underutilized"] = (
        state["memory_stable"]
        & state["cpu_condition_current"]
        & state["cpu_condition_previous"].fillna(False)
    )
    state["gpu_underutilized"] = (
        state["memory_stable"]
        & state["gpu_condition_current"]
        & state["gpu_condition_previous"].fillna(False)
    )

    state["is_underutilized"] = state["cpu_underutilized"] | state["gpu_underutilized"]

    detections = state[
        [
            "timestamp",
            "node_id",
            "cpu_util",
            "gpu_util",
            "memory_util",
            "previous_cpu_util",
            "previous_gpu_util",
            "previous_memory_util",
            "allocated_gpus",
            "previous_allocated_gpus",
            "telemetry_present_any",
            "previous_telemetry_present_any",
            "has_previous_interval",
            "cpu_condition_current",
            "cpu_condition_previous",
            "gpu_condition_current",
            "gpu_condition_previous",
            "memory_condition_current",
            "memory_condition_previous",
            "memory_spike_detected",
            "memory_stable",
            "cpu_underutilized",
            "gpu_underutilized",
            "is_underutilized",
        ]
    ].copy()
    detections["threshold"] = threshold
    detections = detections[
        [
            "timestamp",
            "node_id",
            "threshold",
            "cpu_util",
            "gpu_util",
            "memory_util",
            "previous_cpu_util",
            "previous_gpu_util",
            "previous_memory_util",
            "allocated_gpus",
            "previous_allocated_gpus",
            "telemetry_present_any",
            "previous_telemetry_present_any",
            "has_previous_interval",
            "cpu_condition_current",
            "cpu_condition_previous",
            "gpu_condition_current",
            "gpu_condition_previous",
            "memory_condition_current",
            "memory_condition_previous",
            "memory_spike_detected",
            "memory_stable",
            "cpu_underutilized",
            "gpu_underutilized",
            "is_underutilized",
        ]
    ].sort_values(["timestamp", "node_id"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "underutilized_nodes.parquet"
    detections.to_parquet(output_path, index=False)

    summary = {
        "status": "complete",
        "threshold": threshold,
        "underutilized_nodes_path": str(output_path),
        "rows_written": int(len(detections)),
        "underutilized_rows": int(detections["is_underutilized"].sum()),
        "cpu_underutilized_rows": int(detections["cpu_underutilized"].sum()),
        "gpu_underutilized_rows": int(detections["gpu_underutilized"].sum()),
        "eligible_rows_with_previous_interval": int(detections["has_previous_interval"].sum()),
        "memory_spike_rows": int(detections["memory_spike_detected"].fillna(False).sum()),
        "coverage_start": None if detections.empty else detections["timestamp"].min().isoformat(),
        "coverage_end": None if detections.empty else detections["timestamp"].max().isoformat(),
        "node_count": int(detections["node_id"].nunique()),
        "timestamp_count": int(detections["timestamp"].nunique()),
        "columns": detections.columns.tolist(),
    }
    summary_path = output_dir / "underutilized_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary
