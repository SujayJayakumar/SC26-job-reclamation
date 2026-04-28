from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hpc_sim.config import load_config, resolve_repo_path


def _time(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce", utc=True)
    return timestamps


def _timedelta_seconds(series: pd.Series) -> pd.Series:
    return (
        pd.to_timedelta(series, errors="coerce").fillna(pd.Timedelta(0)) / pd.Timedelta(seconds=1)
    )


def main() -> None:
    root = ROOT
    config = load_config(root / "config" / "default.json")
    processed_dir = resolve_repo_path(config, config["processed_data_dir"])
    results_dir = resolve_repo_path(config, config["results_dir"])

    phase2_summary = json.loads((processed_dir / "phase_02_summary.json").read_text(encoding="utf-8"))
    study_start = pd.Timestamp(phase2_summary["common_overlap_window"]["start"], tz="UTC")
    study_end = pd.Timestamp(phase2_summary["common_overlap_window"]["end"], tz="UTC")

    baseline_payload = json.loads(
        (results_dir / "original_threshold_0.15" / "phase_10_metrics" / "payload.json").read_text(encoding="utf-8")
    )
    baseline_log = pd.read_parquet(
        results_dir / "original_threshold_0.15" / "phase_09_simulation_loop" / "simulation_log.parquet",
        columns=["timestamp", "node_id", "cpu_util", "gpu_util", "memory_util"],
    )

    jobs = pd.read_parquet(
        processed_dir / "job_raw.parquet",
        columns=[
            "job_id",
            "euser",
            "qtime_ts",
            "job_latest_ts",
            "Resource_List.ncpus",
            "Resource_List.ngpus",
            "Resource_List.walltime",
            "resources_used.walltime",
        ],
    )
    jobs["qtime_ts"] = _time(jobs["qtime_ts"])
    jobs["job_latest_ts"] = _time(jobs["job_latest_ts"])
    jobs["requested_cores"] = pd.to_numeric(jobs["Resource_List.ncpus"], errors="coerce").fillna(0.0)
    jobs["requested_gpus"] = pd.to_numeric(jobs["Resource_List.ngpus"], errors="coerce").fillna(0.0)
    jobs["requested_runtime_seconds"] = _timedelta_seconds(jobs["Resource_List.walltime"])
    jobs["observed_runtime_seconds"] = _timedelta_seconds(jobs["resources_used.walltime"])
    overlap_jobs = jobs.loc[(jobs["qtime_ts"] <= study_end) & (jobs["job_latest_ts"] >= study_start)].copy()

    baseline_state_like = pd.read_parquet(
        results_dir / "original_threshold_0.15" / "phase_09_simulation_loop" / "simulation_log.parquet",
        columns=["cpu_util", "gpu_util", "memory_util", "timestamp", "node_id"],
    )

    matched_rows: dict[str, dict[str, float]] = {}
    base = pd.read_parquet(
        results_dir / "original_threshold_0.15" / "phase_09_simulation_loop" / "simulation_log.parquet",
        columns=["timestamp", "node_id", "cpu_util"],
    )
    for mode in ("aggressive", "buffered"):
        sim = pd.read_parquet(
            results_dir / f"{mode}_threshold_0.15" / "phase_09_simulation_loop" / "simulation_log.parquet",
            columns=["timestamp", "node_id", "total_util"],
        )
        matched = sim.merge(base, on=["timestamp", "node_id"], how="left")
        matched_rows[mode] = {
            "matched_baseline_cpu_util_mean": float(matched["cpu_util"].mean()),
            "scenario_total_cpu_util_mean": float(matched["total_util"].mean()),
            "matched_rows": int(len(matched)),
        }

    summary = {
        "study_window": {
            "start": study_start.isoformat(),
            "end": study_end.isoformat(),
            "duration_days": float(phase2_summary["common_overlap_window"]["duration_days"]),
            "node_count": int(baseline_log["node_id"].nunique()),
            "timestamp_count": int(baseline_log["timestamp"].nunique()),
        },
        "baseline_cluster_utilization": {
            "cpu_util_mean": float(baseline_log["cpu_util"].mean()),
            "gpu_util_mean": float(baseline_log["gpu_util"].mean()),
            "memory_util_mean": float(baseline_log["memory_util"].mean()),
        },
        "baseline_state_like_utilization": {
            "cpu_util_mean": float(baseline_state_like["cpu_util"].mean()),
            "gpu_util_mean": float(baseline_state_like["gpu_util"].mean()),
            "memory_util_mean": float(baseline_state_like["memory_util"].mean()),
            "rows": int(len(baseline_state_like)),
        },
        "matched_baseline_example_threshold_015": matched_rows,
        "job_population": {
            "total_jobs_raw": int(jobs["job_id"].nunique()),
            "jobs_overlapping_study_window": int(overlap_jobs["job_id"].nunique()),
            "unique_users_overlapping_window": int(overlap_jobs["euser"].dropna().nunique()),
            "avg_requested_cores_per_job": float(overlap_jobs["requested_cores"].mean()),
            "median_requested_cores_per_job": float(overlap_jobs["requested_cores"].median()),
            "p90_requested_cores_per_job": float(overlap_jobs["requested_cores"].quantile(0.9)),
            "gpu_job_fraction_pct": float((overlap_jobs["requested_gpus"] > 0).mean() * 100.0),
            "avg_requested_gpus_all_jobs": float(overlap_jobs["requested_gpus"].mean()),
            "avg_requested_gpus_gpu_jobs": float(overlap_jobs.loc[overlap_jobs["requested_gpus"] > 0, "requested_gpus"].mean()),
            "avg_observed_runtime_hours": float(overlap_jobs["observed_runtime_seconds"].mean() / 3600.0),
            "avg_requested_runtime_hours": float(overlap_jobs["requested_runtime_seconds"].mean() / 3600.0),
        },
        "notes": {
            "avg_util_per_job": (
                "Per-job CPU/GPU utilization is not directly observable from the current trace because telemetry is node-based, "
                "not job-attributed. The summary therefore reports cluster-level means, baseline state-like row means, and job request sizes."
            ),
            "matched_baseline_interpretation": (
                "Matched baseline CPU utilization is computed only on the node-time slices touched by a non-baseline scenario. "
                "It is therefore much lower than the whole-cluster baseline mean and should not be interpreted as the global cluster average."
            ),
        },
    }

    output_path = processed_dir / "baseline_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
