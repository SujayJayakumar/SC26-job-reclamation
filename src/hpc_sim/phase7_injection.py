from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _load_phase_payload(phase_dir: Path) -> dict:
    payload_path = phase_dir / "payload.json"
    with payload_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_phase_07(
    phase_06_dir: Path,
    scenario_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    phase_06_payload = _load_phase_payload(phase_06_dir)
    phase_03_payload = _load_phase_payload(scenario_dir / "phase_03_state_construction")

    opportunities = pd.read_parquet(
        Path(phase_06_payload["opportunistic_job_pool_path"]),
        columns=[
            "timestamp",
            "node_id",
            "threshold",
            "reclaimable_cores",
            "reclaimable_gpus",
            "has_candidate_fit",
            "opportunistic_job_id",
            "opportunistic_queue",
            "candidate_qtime_ts",
            "candidate_stime_ts",
            "cpu_required",
            "gpu_required",
            "requested_walltime",
            "duration_remaining_seconds",
            "candidate_rank_in_window",
            "fit_cpu_slack",
            "fit_gpu_slack",
        ],
    )
    opportunities["timestamp"] = pd.to_datetime(opportunities["timestamp"])
    opportunities["candidate_qtime_ts"] = pd.to_datetime(opportunities["candidate_qtime_ts"])
    opportunities["candidate_stime_ts"] = pd.to_datetime(opportunities["candidate_stime_ts"])

    state = pd.read_parquet(
        Path(phase_03_payload["state_cache_path"]),
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
            "telemetry_present_any",
        ],
    )
    state["timestamp"] = pd.to_datetime(state["timestamp"])

    injection = opportunities.merge(
        state,
        on=["timestamp", "node_id"],
        how="left",
        suffixes=("", "_primary"),
    )
    injection["opportunistic_job_present_before"] = False
    injection["enough_reclaimable_resources"] = (
        injection["has_candidate_fit"].fillna(False)
        & (injection["cpu_required"].fillna(float("inf")) <= injection["reclaimable_cores"].fillna(-1))
        & (injection["gpu_required"].fillna(float("inf")) <= injection["reclaimable_gpus"].fillna(-1))
    )
    injection["should_inject"] = (
        injection["enough_reclaimable_resources"]
        & ~injection["opportunistic_job_present_before"]
    )

    injection["opportunistic_cpu_allocated"] = injection["cpu_required"].where(
        injection["should_inject"], 0.0
    ).fillna(0.0)
    injection["opportunistic_gpu_allocated"] = injection["gpu_required"].where(
        injection["should_inject"], 0.0
    ).fillna(0.0)
    injection["opportunistic_duration_remaining_seconds"] = injection[
        "duration_remaining_seconds"
    ].where(injection["should_inject"])
    injection["assigned_opportunistic_job_id"] = injection["opportunistic_job_id"].where(
        injection["should_inject"]
    ).astype("string")

    injection["post_injection_allocated_cores"] = (
        injection["allocated_cores"].fillna(0.0)
        + injection["opportunistic_cpu_allocated"].fillna(0.0)
    )
    injection["post_injection_allocated_gpus"] = (
        injection["allocated_gpus"].fillna(0.0)
        + injection["opportunistic_gpu_allocated"].fillna(0.0)
    )
    injection["post_injection_active_job_count"] = (
        injection["active_job_count"].fillna(0).astype("int64")
        + injection["should_inject"].astype("int64")
    )
    injection["node_has_opportunistic_job"] = injection["should_inject"]

    injection_events = injection[
        [
            "timestamp",
            "node_id",
            "threshold",
            "reclaimable_cores",
            "reclaimable_gpus",
            "has_candidate_fit",
            "enough_reclaimable_resources",
            "opportunistic_job_present_before",
            "should_inject",
            "assigned_opportunistic_job_id",
            "opportunistic_queue",
            "candidate_qtime_ts",
            "candidate_stime_ts",
            "opportunistic_cpu_allocated",
            "opportunistic_gpu_allocated",
            "requested_walltime",
            "opportunistic_duration_remaining_seconds",
            "candidate_rank_in_window",
            "fit_cpu_slack",
            "fit_gpu_slack",
        ]
    ].sort_values(["timestamp", "node_id"]).reset_index(drop=True)

    state_updates = injection[
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
            "node_has_opportunistic_job",
            "assigned_opportunistic_job_id",
            "opportunistic_cpu_allocated",
            "opportunistic_gpu_allocated",
            "opportunistic_duration_remaining_seconds",
            "post_injection_allocated_cores",
            "post_injection_allocated_gpus",
            "post_injection_active_job_count",
        ]
    ].sort_values(["timestamp", "node_id"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "injection_events.parquet"
    state_path = output_dir / "state_after_injection.parquet"
    injection_events.to_parquet(events_path, index=False)
    state_updates.to_parquet(state_path, index=False)

    summary = {
        "status": "complete",
        "injection_events_path": str(events_path),
        "state_after_injection_path": str(state_path),
        "rows_written": int(len(injection_events)),
        "coverage_start": None if injection_events.empty else injection_events["timestamp"].min().isoformat(),
        "coverage_end": None if injection_events.empty else injection_events["timestamp"].max().isoformat(),
        "node_count": int(injection_events["node_id"].nunique()) if not injection_events.empty else 0,
        "timestamp_count": int(injection_events["timestamp"].nunique()) if not injection_events.empty else 0,
        "rows_with_candidate_fit": int(injection_events["has_candidate_fit"].sum()) if not injection_events.empty else 0,
        "successful_injections": int(injection_events["should_inject"].sum()) if not injection_events.empty else 0,
        "rows_without_injection": int((~injection_events["should_inject"]).sum()) if not injection_events.empty else 0,
        "unique_injected_jobs": int(injection_events["assigned_opportunistic_job_id"].dropna().nunique()) if not injection_events.empty else 0,
        "max_post_injection_allocated_cores": float(state_updates["post_injection_allocated_cores"].max()) if not state_updates.empty else 0.0,
        "max_post_injection_allocated_gpus": float(state_updates["post_injection_allocated_gpus"].max()) if not state_updates.empty else 0.0,
        "columns": injection_events.columns.tolist(),
        "state_columns": state_updates.columns.tolist(),
        "injection_rule": (
            "inject best-fit opportunistic job when candidate fits reclaimable resources and "
            "no opportunistic job is already present on that node-slot"
        ),
    }
    summary_path = output_dir / "injection_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary
