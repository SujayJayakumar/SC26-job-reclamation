from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


CPU_CORES_PER_NODE = 128.0


def _load_phase_payload(phase_dir: Path) -> dict:
    payload_path = phase_dir / "payload.json"
    with payload_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _buffer_total_allowed(config: dict, threshold: float) -> float:
    key = f"{threshold:.2f}"
    return float(config["buffers"][key])


def run_phase_08(
    config: dict,
    phase_07_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    phase_07_payload = _load_phase_payload(phase_07_dir)
    injection_events = pd.read_parquet(
        Path(phase_07_payload["injection_events_path"]),
        columns=[
            "timestamp",
            "node_id",
            "threshold",
            "should_inject",
            "assigned_opportunistic_job_id",
            "opportunistic_cpu_allocated",
            "opportunistic_gpu_allocated",
            "opportunistic_duration_remaining_seconds",
        ],
    )
    state_after_injection = pd.read_parquet(
        Path(phase_07_payload["state_after_injection_path"]),
        columns=[
            "timestamp",
            "node_id",
            "cpu_util",
            "gpu_util",
            "memory_util",
            "job_id",
            "active_job_count",
            "allocated_cores",
            "allocated_gpus",
            "telemetry_present_any",
            "node_has_opportunistic_job",
            "assigned_opportunistic_job_id",
            "opportunistic_cpu_allocated",
            "opportunistic_gpu_allocated",
            "opportunistic_duration_remaining_seconds",
            "post_injection_allocated_cores",
            "post_injection_allocated_gpus",
            "post_injection_active_job_count",
        ],
    )
    injection_events["timestamp"] = pd.to_datetime(injection_events["timestamp"])
    state_after_injection["timestamp"] = pd.to_datetime(state_after_injection["timestamp"])

    preemption = state_after_injection.merge(
        injection_events[
            [
                "timestamp",
                "node_id",
                "threshold",
                "should_inject",
            ]
        ],
        on=["timestamp", "node_id"],
        how="left",
    )
    preemption["threshold"] = preemption["threshold"].astype(float)
    preemption["allowed_total_util"] = preemption["threshold"].map(
        lambda value: _buffer_total_allowed(config, value)
    )
    preemption["opportunistic_cpu_util_estimate"] = (
        preemption["opportunistic_cpu_allocated"].fillna(0.0) / CPU_CORES_PER_NODE
    )
    preemption["total_util"] = (
        preemption["cpu_util"].fillna(0.0)
        + preemption["opportunistic_cpu_util_estimate"]
    )
    preemption["preempt_opportunistic_job"] = (
        preemption["node_has_opportunistic_job"].fillna(False)
        & (preemption["total_util"] > preemption["allowed_total_util"])
    )

    preemption["restored_cpu_cores"] = preemption["opportunistic_cpu_allocated"].where(
        preemption["preempt_opportunistic_job"], 0.0
    ).fillna(0.0)
    preemption["restored_gpu_gpus"] = preemption["opportunistic_gpu_allocated"].where(
        preemption["preempt_opportunistic_job"], 0.0
    ).fillna(0.0)
    preemption["remaining_opportunistic_job_id"] = preemption[
        "assigned_opportunistic_job_id"
    ].where(~preemption["preempt_opportunistic_job"]).astype("string")
    preemption["remaining_opportunistic_cpu_allocated"] = preemption[
        "opportunistic_cpu_allocated"
    ].where(~preemption["preempt_opportunistic_job"], 0.0).fillna(0.0)
    preemption["remaining_opportunistic_gpu_allocated"] = preemption[
        "opportunistic_gpu_allocated"
    ].where(~preemption["preempt_opportunistic_job"], 0.0).fillna(0.0)
    preemption["remaining_opportunistic_duration_seconds"] = preemption[
        "opportunistic_duration_remaining_seconds"
    ].where(~preemption["preempt_opportunistic_job"])

    preemption["post_preemption_allocated_cores"] = (
        preemption["post_injection_allocated_cores"].fillna(0.0)
        - preemption["restored_cpu_cores"]
    )
    preemption["post_preemption_allocated_gpus"] = (
        preemption["post_injection_allocated_gpus"].fillna(0.0)
        - preemption["restored_gpu_gpus"]
    )
    preemption["post_preemption_active_job_count"] = (
        preemption["post_injection_active_job_count"].fillna(0).astype("int64")
        - preemption["preempt_opportunistic_job"].astype("int64")
    )
    preemption["node_has_opportunistic_job_after_preemption"] = (
        preemption["node_has_opportunistic_job"].fillna(False)
        & ~preemption["preempt_opportunistic_job"]
    )

    preemption_events = preemption[
        [
            "timestamp",
            "node_id",
            "threshold",
            "cpu_util",
            "gpu_util",
            "memory_util",
            "allowed_total_util",
            "opportunistic_cpu_util_estimate",
            "total_util",
            "node_has_opportunistic_job",
            "preempt_opportunistic_job",
            "assigned_opportunistic_job_id",
            "remaining_opportunistic_job_id",
            "restored_cpu_cores",
            "restored_gpu_gpus",
            "remaining_opportunistic_cpu_allocated",
            "remaining_opportunistic_gpu_allocated",
            "opportunistic_duration_remaining_seconds",
            "remaining_opportunistic_duration_seconds",
        ]
    ].sort_values(["timestamp", "node_id"]).reset_index(drop=True)

    state_after_preemption = preemption[
        [
            "timestamp",
            "node_id",
            "cpu_util",
            "gpu_util",
            "memory_util",
            "job_id",
            "active_job_count",
            "allocated_cores",
            "allocated_gpus",
            "telemetry_present_any",
            "node_has_opportunistic_job_after_preemption",
            "remaining_opportunistic_job_id",
            "remaining_opportunistic_cpu_allocated",
            "remaining_opportunistic_gpu_allocated",
            "remaining_opportunistic_duration_seconds",
            "post_preemption_allocated_cores",
            "post_preemption_allocated_gpus",
            "post_preemption_active_job_count",
        ]
    ].sort_values(["timestamp", "node_id"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "preemption_events.parquet"
    state_path = output_dir / "state_after_preemption.parquet"
    preemption_events.to_parquet(events_path, index=False)
    state_after_preemption.to_parquet(state_path, index=False)

    summary = {
        "status": "complete",
        "preemption_events_path": str(events_path),
        "state_after_preemption_path": str(state_path),
        "rows_written": int(len(preemption_events)),
        "coverage_start": None if preemption_events.empty else preemption_events["timestamp"].min().isoformat(),
        "coverage_end": None if preemption_events.empty else preemption_events["timestamp"].max().isoformat(),
        "node_count": int(preemption_events["node_id"].nunique()) if not preemption_events.empty else 0,
        "timestamp_count": int(preemption_events["timestamp"].nunique()) if not preemption_events.empty else 0,
        "rows_with_opportunistic_job_before_preemption": int(preemption_events["node_has_opportunistic_job"].sum()) if not preemption_events.empty else 0,
        "preemption_count": int(preemption_events["preempt_opportunistic_job"].sum()) if not preemption_events.empty else 0,
        "rows_with_opportunistic_job_after_preemption": int(state_after_preemption["node_has_opportunistic_job_after_preemption"].sum()) if not state_after_preemption.empty else 0,
        "max_total_util": float(preemption_events["total_util"].max()) if not preemption_events.empty else 0.0,
        "max_allowed_total_util": float(preemption_events["allowed_total_util"].max()) if not preemption_events.empty else 0.0,
        "columns": preemption_events.columns.tolist(),
        "state_columns": state_after_preemption.columns.tolist(),
        "preemption_rule": (
            "preempt opportunistic job when total cpu utilization estimate "
            "(primary cpu_util + opportunistic_cpu_allocated/128) exceeds "
            "the scenario's total allowed utilization ceiling"
        ),
    }
    summary_path = output_dir / "preemption_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary
