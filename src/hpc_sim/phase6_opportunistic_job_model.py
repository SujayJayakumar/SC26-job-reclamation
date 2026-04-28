from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve_repo_path


FAR_FUTURE_TS = pd.Timestamp.max.floor("s")


def _load_phase_payload(phase_dir: Path) -> dict:
    payload_path = phase_dir / "payload.json"
    with payload_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _time_basis(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce")
    if hasattr(timestamps.dt, "tz") and timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert("UTC").dt.tz_localize(None)
    return timestamps


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _resolve_runtime_seconds(jobs: pd.DataFrame) -> pd.Series:
    requested = (
        pd.to_timedelta(jobs["Resource_List.walltime"], errors="coerce").fillna(pd.Timedelta(0))
        / pd.Timedelta(seconds=1)
    )
    observed_walltime = (
        pd.to_timedelta(jobs["resources_used.walltime"], errors="coerce").fillna(pd.Timedelta(0))
        / pd.Timedelta(seconds=1)
    )
    observed_from_timestamps = (
        (_time_basis(jobs["etime_ts"]) - _time_basis(jobs["stime_ts"])) / pd.Timedelta(seconds=1)
    ).fillna(0.0)
    runtime = observed_walltime.where(observed_walltime > 0, observed_from_timestamps)
    runtime = runtime.where(runtime > 0, requested)
    return runtime.fillna(0.0).round().astype("int64")


def _load_queue_jobs(config: dict, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    processed_dir = resolve_repo_path(config, config["processed_data_dir"])
    job_raw_path = processed_dir / "job_raw.parquet"
    jobs = pd.read_parquet(
        job_raw_path,
        columns=[
            "job_id",
            "queue",
            "qtime_ts",
            "stime_ts",
            "etime_ts",
            "Resource_List.ncpus",
            "Resource_List.ngpus",
            "Resource_List.walltime",
            "resources_used.walltime",
        ],
    )
    jobs["qtime_ts"] = _time_basis(jobs["qtime_ts"])
    jobs["stime_ts"] = _time_basis(jobs["stime_ts"])
    jobs["cpu_required"] = _safe_numeric(jobs["Resource_List.ncpus"]).fillna(0.0)
    jobs["gpu_required"] = _safe_numeric(jobs["Resource_List.ngpus"]).fillna(0.0)
    jobs["requested_walltime"] = pd.to_timedelta(jobs["Resource_List.walltime"], errors="coerce")
    jobs["requested_walltime"] = jobs["requested_walltime"].fillna(pd.Timedelta(0))
    jobs["duration_remaining_seconds"] = _resolve_runtime_seconds(jobs)

    jobs = jobs.loc[
        jobs["qtime_ts"].notna()
        & (jobs["qtime_ts"] >= window_start)
        & (jobs["qtime_ts"] <= window_end)
        & (
            jobs["stime_ts"].isna()
            | (jobs["stime_ts"] > window_start)
        )
        & ((jobs["cpu_required"] > 0) | (jobs["gpu_required"] > 0))
    ].copy()
    jobs["queue"] = jobs["queue"].astype("string")
    jobs["job_id"] = jobs["job_id"].astype("string")
    jobs = jobs.sort_values(["qtime_ts", "job_id"]).reset_index(drop=True)
    return jobs[
        [
            "job_id",
            "queue",
            "qtime_ts",
            "stime_ts",
            "cpu_required",
            "gpu_required",
            "requested_walltime",
            "duration_remaining_seconds",
        ]
    ]


def _build_waiting_windows(jobs: pd.DataFrame, timestamps: pd.Series, window_size: int) -> dict[pd.Timestamp, list[int]]:
    if jobs.empty or timestamps.empty:
        return {}

    queue_times = jobs["qtime_ts"].to_numpy(dtype="datetime64[ns]")
    start_times = jobs["stime_ts"].fillna(FAR_FUTURE_TS).to_numpy(dtype="datetime64[ns]")
    sorted_start_indices = np.argsort(start_times, kind="stable")
    sorted_start_times = start_times[sorted_start_indices]
    timeline = pd.to_datetime(timestamps).sort_values().drop_duplicates().tolist()

    waiting_active = np.zeros(len(jobs), dtype=bool)
    waiting_order: list[int] = []
    windows: dict[pd.Timestamp, list[int]] = {}

    q_ptr = 0
    s_ptr = 0
    for timestamp in timeline:
        timestamp64 = np.datetime64(timestamp.to_datetime64(), "ns")
        while q_ptr < len(queue_times) and queue_times[q_ptr] <= timestamp64:
            waiting_active[q_ptr] = True
            waiting_order.append(q_ptr)
            q_ptr += 1

        while s_ptr < len(sorted_start_times) and sorted_start_times[s_ptr] <= timestamp64:
            waiting_active[sorted_start_indices[s_ptr]] = False
            s_ptr += 1

        top_indices: list[int] = []
        for idx in waiting_order:
            if waiting_active[idx]:
                top_indices.append(idx)
                if len(top_indices) >= window_size:
                    break
        windows[timestamp] = top_indices

    return windows


def _select_best_fit(
    node_rows: pd.DataFrame,
    queue_window: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    queue_rows = list(queue_window.itertuples(index=False))
    window_size_actual = len(queue_rows)

    for row in node_rows.itertuples(index=False):
        reclaimable_cores = float(row.reclaimable_cores)
        reclaimable_gpus = float(row.reclaimable_gpus)
        best_record: dict[str, object] | None = None
        fit_count = 0

        for rank, candidate in enumerate(queue_rows, start=1):
            cpu_required = float(candidate.cpu_required)
            gpu_required = float(candidate.gpu_required)
            cpu_slack = reclaimable_cores - cpu_required
            gpu_slack = reclaimable_gpus - gpu_required
            if cpu_slack < 0 or gpu_slack < 0:
                continue

            fit_count += 1
            candidate_record = {
                "opportunistic_job_id": candidate.job_id,
                "opportunistic_queue": candidate.queue,
                "candidate_qtime_ts": candidate.qtime_ts,
                "candidate_stime_ts": candidate.stime_ts,
                "cpu_required": cpu_required,
                "gpu_required": gpu_required,
                "requested_walltime": candidate.requested_walltime,
                "duration_remaining_seconds": int(candidate.duration_remaining_seconds),
                "candidate_rank_in_window": rank,
                "fit_cpu_slack": int(cpu_slack),
                "fit_gpu_slack": int(gpu_slack),
            }
            if best_record is None:
                best_record = candidate_record
                continue

            current_key = (
                candidate_record["fit_gpu_slack"],
                candidate_record["fit_cpu_slack"],
                candidate_record["candidate_rank_in_window"],
                candidate_record["duration_remaining_seconds"],
                str(candidate_record["opportunistic_job_id"]),
            )
            best_key = (
                best_record["fit_gpu_slack"],
                best_record["fit_cpu_slack"],
                best_record["candidate_rank_in_window"],
                best_record["duration_remaining_seconds"],
                str(best_record["opportunistic_job_id"]),
            )
            if current_key < best_key:
                best_record = candidate_record

        records.append(
            {
                "timestamp": row.timestamp,
                "node_id": row.node_id,
                "threshold": row.threshold,
                "reclaimable_cores": int(reclaimable_cores),
                "reclaimable_gpus": int(reclaimable_gpus),
                "selection_window_size": window_size_actual,
                "fit_candidate_count": fit_count,
                "has_candidate_fit": best_record is not None,
                **(
                    best_record
                    if best_record is not None
                    else {
                        "opportunistic_job_id": pd.NA,
                        "opportunistic_queue": pd.NA,
                        "candidate_qtime_ts": pd.NaT,
                        "candidate_stime_ts": pd.NaT,
                        "cpu_required": pd.NA,
                        "gpu_required": pd.NA,
                        "requested_walltime": pd.NaT,
                        "duration_remaining_seconds": pd.NA,
                        "candidate_rank_in_window": pd.NA,
                        "fit_cpu_slack": pd.NA,
                        "fit_gpu_slack": pd.NA,
                    }
                ),
            }
        )

    return pd.DataFrame.from_records(records)


def run_phase_06(
    config: dict,
    phase_05_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    phase_05_payload = _load_phase_payload(phase_05_dir)
    reclamation_path = Path(phase_05_payload["reclamation_plan_path"])
    reclamation = pd.read_parquet(
        reclamation_path,
        columns=[
            "timestamp",
            "node_id",
            "threshold",
            "reclaimable_cores",
            "reclaimable_gpus",
        ],
    )
    reclamation["timestamp"] = pd.to_datetime(reclamation["timestamp"])
    reclamation = reclamation.sort_values(["timestamp", "node_id"]).reset_index(drop=True)

    if reclamation.empty:
        opportunities = pd.DataFrame(
            columns=[
                "timestamp",
                "node_id",
                "threshold",
                "reclaimable_cores",
                "reclaimable_gpus",
                "selection_window_size",
                "fit_candidate_count",
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
            ]
        )
    else:
        queue_window_size = int(config["queue"]["selection_window"])
        jobs = _load_queue_jobs(
            config,
            window_start=reclamation["timestamp"].min(),
            window_end=reclamation["timestamp"].max(),
        )
        waiting_windows = _build_waiting_windows(
            jobs=jobs,
            timestamps=reclamation["timestamp"],
            window_size=queue_window_size,
        )

        batches: list[pd.DataFrame] = []
        for timestamp, node_rows in reclamation.groupby("timestamp", sort=True):
            queue_indices = waiting_windows.get(timestamp, [])
            queue_window = jobs.iloc[queue_indices].reset_index(drop=True)
            selected = _select_best_fit(node_rows.reset_index(drop=True), queue_window)
            batches.append(selected)
        opportunities = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "opportunistic_job_pool.parquet"
    opportunities.to_parquet(output_path, index=False)

    summary = {
        "status": "complete",
        "opportunistic_job_pool_path": str(output_path),
        "rows_written": int(len(opportunities)),
        "coverage_start": None if opportunities.empty else opportunities["timestamp"].min().isoformat(),
        "coverage_end": None if opportunities.empty else opportunities["timestamp"].max().isoformat(),
        "node_count": int(opportunities["node_id"].nunique()) if not opportunities.empty else 0,
        "timestamp_count": int(opportunities["timestamp"].nunique()) if not opportunities.empty else 0,
        "rows_with_candidate_fit": int(opportunities["has_candidate_fit"].sum()) if not opportunities.empty else 0,
        "rows_without_candidate_fit": int((~opportunities["has_candidate_fit"]).sum()) if not opportunities.empty else 0,
        "unique_selected_jobs": int(opportunities["opportunistic_job_id"].dropna().nunique()) if not opportunities.empty else 0,
        "max_selection_window_size": int(opportunities["selection_window_size"].max()) if not opportunities.empty else 0,
        "max_fit_candidate_count": int(opportunities["fit_candidate_count"].max()) if not opportunities.empty else 0,
        "columns": opportunities.columns.tolist(),
        "selection_policy": (
            "top-10 waiting queue snapshot at each timestamp; choose best-fit job with "
            "minimum gpu slack, then minimum cpu slack, then earliest queue rank"
        ),
        "duration_remaining_rule": (
            "initialized from requested walltime for waiting jobs; later phases should decrement "
            "after execution and preserve remaining duration across preemption"
        ),
    }
    summary_path = output_dir / "opportunistic_job_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary
