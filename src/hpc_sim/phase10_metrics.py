from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


CPU_CORES_PER_NODE = 128.0
INTERVAL_MINUTES = 10.0
SCENARIO_RE = re.compile(r"^(?P<mode>[a-z]+)_threshold_(?P<threshold>\d+\.\d+)$")


def _load_phase_payload(phase_dir: Path) -> dict:
    payload_path = phase_dir / "payload.json"
    with payload_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_scenario(scenario: str) -> tuple[str, float]:
    match = SCENARIO_RE.match(scenario)
    if not match:
        raise ValueError(f"Unable to parse scenario: {scenario}")
    return match.group("mode"), float(match.group("threshold"))


def _normalize_util(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric.dropna()
    if finite.empty:
        return numeric
    if float(finite.quantile(0.99)) > 1.5:
        return numeric / 100.0
    return numeric


def _week_start(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce")
    return timestamps.dt.to_period("W-SUN").dt.start_time


def _baseline_summary(results_root: Path, threshold: float) -> dict[str, object]:
    baseline_scenario = f"original_threshold_{threshold:.2f}"
    phase_09_payload = _load_phase_payload(results_root / baseline_scenario / "phase_09_simulation_loop")
    baseline_path = Path(phase_09_payload["simulation_log_path"])
    if baseline_path.exists():
        baseline = pd.read_parquet(
            baseline_path,
            columns=["timestamp", "node_id", "cpu_util", "gpu_util", "total_util"],
        )
        baseline["cpu_util"] = _normalize_util(baseline["cpu_util"])
        baseline["gpu_util"] = _normalize_util(baseline["gpu_util"])
        baseline["total_util"] = _normalize_util(baseline["total_util"])
        baseline["timestamp"] = pd.to_datetime(baseline["timestamp"])
    else:
        phase_03_dir = results_root / baseline_scenario / "phase_03_state_construction"
        phase_03_payload = _load_phase_payload(phase_03_dir)
        state_cache_path = Path(phase_03_payload["state_cache_path"])
        baseline = pd.read_parquet(
            state_cache_path,
            columns=["timestamp", "node_id", "cpu_util", "gpu_util"],
        )
        baseline["timestamp"] = pd.to_datetime(baseline["timestamp"])
        baseline["cpu_util"] = _normalize_util(baseline["cpu_util"])
        baseline["gpu_util"] = _normalize_util(baseline["gpu_util"])
        baseline["total_util"] = baseline["cpu_util"].fillna(0.0)

    node_count = int(baseline["node_id"].nunique())
    timestamp_count = int(baseline["timestamp"].nunique())
    capacity_core_minutes = node_count * timestamp_count * CPU_CORES_PER_NODE * INTERVAL_MINUTES
    primary_core_minutes = float((baseline["cpu_util"].fillna(0.0) * CPU_CORES_PER_NODE * INTERVAL_MINUTES).sum())

    return {
        "scenario": baseline_scenario,
        "node_count": node_count,
        "timestamp_count": timestamp_count,
        "capacity_core_minutes": capacity_core_minutes,
        "baseline_primary_core_minutes": primary_core_minutes,
        "baseline_mean_cpu_util": float(baseline["cpu_util"].fillna(0.0).mean()),
        "baseline_mean_gpu_util": float(baseline["gpu_util"].fillna(0.0).mean()),
    }


def _build_weekly_metrics(log: pd.DataFrame) -> pd.DataFrame:
    if log.empty:
        return pd.DataFrame(
            columns=[
                "week_start",
                "rows",
                "active_opportunistic_rows",
                "opportunistic_cpu_core_minutes",
                "opportunistic_gpu_device_minutes",
                "mean_primary_cpu_util",
                "mean_total_cpu_util",
                "cpu_util_improvement_points",
                "preemption_count",
                "interference_event_count",
                "completion_count",
                "unique_opportunistic_jobs_run",
            ]
        )

    log = log.copy()
    log["week_start"] = _week_start(log["timestamp"])
    active_indicator = (
        pd.to_numeric(log["opportunistic_job_count"], errors="coerce").fillna(0)
        if "opportunistic_job_count" in log.columns
        else log["opportunistic_job_id"].notna().astype("int64")
    )
    log["active_opportunistic_indicator"] = active_indicator
    grouped = log.groupby("week_start", as_index=False)
    weekly = grouped.agg(
        rows=("node_id", "size"),
        active_opportunistic_rows=("active_opportunistic_indicator", lambda values: int(values.fillna(0).sum())),
        opportunistic_cpu_core_minutes=("opportunistic_cpu_allocated", lambda values: float(values.fillna(0.0).sum() * INTERVAL_MINUTES)),
        opportunistic_gpu_device_minutes=("opportunistic_gpu_allocated", lambda values: float(values.fillna(0.0).sum() * INTERVAL_MINUTES)),
        mean_primary_cpu_util=("cpu_util", lambda values: float(values.fillna(0.0).mean())),
        mean_total_cpu_util=("total_util", lambda values: float(values.fillna(0.0).mean())),
        preemption_count=("preempted_this_interval", lambda values: int(values.fillna(False).sum())),
        interference_event_count=("preempted_this_interval", lambda values: int(values.fillna(False).sum())),
        completion_count=("completed_this_interval", lambda values: int(values.fillna(False).sum())),
        unique_opportunistic_jobs_run=("opportunistic_job_id", lambda values: int(values.dropna().nunique())),
    )
    weekly["cpu_util_improvement_points"] = weekly["mean_total_cpu_util"] - weekly["mean_primary_cpu_util"]
    return weekly.sort_values("week_start").reset_index(drop=True)


def run_phase_10(config: dict, input_dir: Path, scenario_dir: Path, output_dir: Path, scenario: str) -> dict[str, object]:
    del config
    mode, threshold = _parse_scenario(scenario)
    results_root = scenario_dir.parent
    phase_09_payload = _load_phase_payload(input_dir)
    baseline = _baseline_summary(results_root, threshold)

    if mode == "original":
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_row = pd.DataFrame(
            [
                {
                    "scenario": scenario,
                    "mode": mode,
                    "threshold": threshold,
                    "allowed_total_util": phase_09_payload["allowed_total_util"],
                    "rows_evaluated": int(phase_09_payload["rows_written"]),
                    "active_opportunistic_rows": 0,
                    "utilization_improvement_points": 0.0,
                    "cluster_utilization_improvement_pct": 0.0,
                    "mean_primary_cpu_util": float(baseline["baseline_mean_cpu_util"]),
                    "mean_total_cpu_util": float(baseline["baseline_mean_cpu_util"]),
                    "preemption_count": 0,
                    "interference_event_count": 0,
                    "completion_count": 0,
                    "throughput_proxy_cpu_core_minutes": 0.0,
                    "throughput_proxy_gpu_device_minutes": 0.0,
                    "relative_cpu_throughput_gain_fraction": 0.0,
                    "relative_cpu_throughput_gain_pct": 0.0,
                    "unique_opportunistic_jobs_run": 0,
                    "baseline_scenario": baseline["scenario"],
                    "baseline_node_count": baseline["node_count"],
                    "baseline_timestamp_count": baseline["timestamp_count"],
                }
            ]
        )
        weekly_metrics = pd.DataFrame(
            columns=[
                "week_start",
                "rows",
                "active_opportunistic_rows",
                "opportunistic_cpu_core_minutes",
                "opportunistic_gpu_device_minutes",
                "mean_primary_cpu_util",
                "mean_total_cpu_util",
                "cpu_util_improvement_points",
                "preemption_count",
                "interference_event_count",
                "completion_count",
                "unique_opportunistic_jobs_run",
            ]
        )
        summary_path = output_dir / "metrics_summary.parquet"
        weekly_path = output_dir / "metrics_weekly.parquet"
        json_path = output_dir / "metrics_summary.json"
        summary_row.to_parquet(summary_path, index=False)
        weekly_metrics.to_parquet(weekly_path, index=False)
        summary = {
            "status": "complete",
            "scenario": scenario,
            "mode": mode,
            "threshold": threshold,
            "allowed_total_util": phase_09_payload["allowed_total_util"],
            "metrics_summary_path": str(summary_path),
            "metrics_weekly_path": str(weekly_path),
            "rows_evaluated": int(phase_09_payload["rows_written"]),
            "active_opportunistic_rows": 0,
            "utilization_improvement_points": 0.0,
            "cluster_utilization_improvement_pct": 0.0,
            "mean_primary_cpu_util": float(baseline["baseline_mean_cpu_util"]),
            "mean_total_cpu_util": float(baseline["baseline_mean_cpu_util"]),
            "preemption_count": 0,
            "interference_event_count": 0,
            "completion_count": 0,
            "throughput_proxy_cpu_core_minutes": 0.0,
            "throughput_proxy_gpu_device_minutes": 0.0,
            "relative_cpu_throughput_gain_fraction": 0.0,
            "relative_cpu_throughput_gain_pct": 0.0,
            "unique_opportunistic_jobs_run": 0,
            "weekly_windows": 0,
            "metric_definitions": {
                "utilization_improvement": "baseline scenario; no opportunistic uplift is applied",
                "preemption_count": "baseline scenario; no opportunistic jobs are injected",
                "interference_events": "baseline scenario; no opportunistic jobs are injected",
                "throughput_proxy": "baseline scenario; no opportunistic jobs are injected",
            },
            "baseline_reference": baseline,
        }
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return summary

    simulation_log_path = Path(phase_09_payload["simulation_log_path"])
    log = pd.read_parquet(simulation_log_path)
    log["timestamp"] = pd.to_datetime(log["timestamp"])
    log["cpu_util"] = _normalize_util(log["cpu_util"])
    log["gpu_util"] = _normalize_util(log["gpu_util"])
    log["total_util"] = _normalize_util(log["total_util"])
    log["opportunistic_cpu_allocated"] = pd.to_numeric(log["opportunistic_cpu_allocated"], errors="coerce").fillna(0.0)
    log["opportunistic_gpu_allocated"] = pd.to_numeric(log["opportunistic_gpu_allocated"], errors="coerce").fillna(0.0)

    if "opportunistic_job_count" in log.columns:
        active_opportunistic_rows = int(pd.to_numeric(log["opportunistic_job_count"], errors="coerce").fillna(0).sum())
    else:
        active_opportunistic_rows = int(log["opportunistic_job_id"].notna().sum()) if "opportunistic_job_id" in log.columns else 0
    opportunistic_cpu_core_minutes = float(log["opportunistic_cpu_allocated"].sum() * INTERVAL_MINUTES)
    opportunistic_gpu_device_minutes = float(log["opportunistic_gpu_allocated"].sum() * INTERVAL_MINUTES)
    mean_primary_cpu_util = float(log["cpu_util"].fillna(0.0).mean()) if not log.empty else 0.0
    mean_total_cpu_util = float(log["total_util"].fillna(0.0).mean()) if not log.empty else 0.0
    cpu_util_improvement_points = mean_total_cpu_util - mean_primary_cpu_util

    preemption_count = int(log["preempted_this_interval"].fillna(False).sum()) if "preempted_this_interval" in log.columns else 0
    interference_event_count = preemption_count
    completion_count = int(log["completed_this_interval"].fillna(False).sum()) if "completed_this_interval" in log.columns else 0
    unique_jobs = int(phase_09_payload.get("unique_opportunistic_jobs_run", 0))

    cluster_capacity_core_minutes = float(baseline["capacity_core_minutes"])
    cluster_utilization_improvement_pct = (
        (opportunistic_cpu_core_minutes / cluster_capacity_core_minutes) * 100.0
        if cluster_capacity_core_minutes > 0
        else 0.0
    )
    baseline_primary_core_minutes = float(baseline["baseline_primary_core_minutes"])
    relative_cpu_throughput_gain_fraction = (
        (opportunistic_cpu_core_minutes / baseline_primary_core_minutes)
        if baseline_primary_core_minutes > 0
        else 0.0
    )
    relative_cpu_throughput_gain_pct = relative_cpu_throughput_gain_fraction * 100.0

    weekly_metrics = _build_weekly_metrics(log)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_row = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "mode": mode,
                "threshold": threshold,
                "allowed_total_util": phase_09_payload["allowed_total_util"],
                "rows_evaluated": int(len(log)),
                "active_opportunistic_rows": active_opportunistic_rows,
                "utilization_improvement_points": cpu_util_improvement_points,
                "cluster_utilization_improvement_pct": cluster_utilization_improvement_pct,
                "mean_primary_cpu_util": mean_primary_cpu_util,
                "mean_total_cpu_util": mean_total_cpu_util,
                "preemption_count": preemption_count,
                "interference_event_count": interference_event_count,
                "completion_count": completion_count,
                "throughput_proxy_cpu_core_minutes": opportunistic_cpu_core_minutes,
                "throughput_proxy_gpu_device_minutes": opportunistic_gpu_device_minutes,
                "relative_cpu_throughput_gain_fraction": relative_cpu_throughput_gain_fraction,
                "relative_cpu_throughput_gain_pct": relative_cpu_throughput_gain_pct,
                "unique_opportunistic_jobs_run": unique_jobs,
                "baseline_scenario": baseline["scenario"],
                "baseline_node_count": baseline["node_count"],
                "baseline_timestamp_count": baseline["timestamp_count"],
            }
        ]
    )

    summary_path = output_dir / "metrics_summary.parquet"
    weekly_path = output_dir / "metrics_weekly.parquet"
    json_path = output_dir / "metrics_summary.json"
    summary_row.to_parquet(summary_path, index=False)
    weekly_metrics.to_parquet(weekly_path, index=False)

    summary = {
        "status": "complete",
        "scenario": scenario,
        "mode": mode,
        "threshold": threshold,
        "allowed_total_util": phase_09_payload["allowed_total_util"],
        "metrics_summary_path": str(summary_path),
        "metrics_weekly_path": str(weekly_path),
        "rows_evaluated": int(len(log)),
        "active_opportunistic_rows": active_opportunistic_rows,
        "utilization_improvement_points": cpu_util_improvement_points,
        "cluster_utilization_improvement_pct": cluster_utilization_improvement_pct,
        "mean_primary_cpu_util": mean_primary_cpu_util,
        "mean_total_cpu_util": mean_total_cpu_util,
        "preemption_count": preemption_count,
        "interference_event_count": interference_event_count,
        "completion_count": completion_count,
        "throughput_proxy_cpu_core_minutes": opportunistic_cpu_core_minutes,
        "throughput_proxy_gpu_device_minutes": opportunistic_gpu_device_minutes,
        "relative_cpu_throughput_gain_fraction": relative_cpu_throughput_gain_fraction,
        "relative_cpu_throughput_gain_pct": relative_cpu_throughput_gain_pct,
        "unique_opportunistic_jobs_run": unique_jobs,
        "weekly_windows": int(len(weekly_metrics)),
        "metric_definitions": {
            "utilization_improvement": "mean(total_util - primary_cpu_util) across the simulated rows, plus whole-cluster CPU capacity uplift from executed opportunistic core-minutes",
            "preemption_count": "count of node-timestamp intervals where opportunistic work was preempted to protect primary workload",
            "interference_events": "same as preemption_count in the current conservative model, because primary interference is represented by protective preemption before oversubscription is allowed to persist",
            "throughput_proxy": "executed opportunistic CPU core-minutes and GPU device-minutes",
            "relative_cpu_throughput_gain_fraction": "opportunistic_cpu_core_minutes / baseline_primary_core_minutes",
            "relative_cpu_throughput_gain_pct": "100 * opportunistic_cpu_core_minutes / baseline_primary_core_minutes",
        },
        "baseline_reference": baseline,
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary
