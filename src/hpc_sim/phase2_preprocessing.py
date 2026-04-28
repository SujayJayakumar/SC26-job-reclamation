from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import resolve_repo_path


def _floor_to_10m(series: pd.Series) -> pd.Series:
    timestamps = _phase2_time_basis(series)
    return timestamps.dt.floor("10min")


def _phase2_time_basis(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce")
    if hasattr(timestamps.dt, "tz") and timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert("UTC").dt.tz_localize(None)
    return timestamps


def _window_start_10m(series: pd.Series) -> pd.Timestamp:
    return _floor_to_10m(series).min()


def _window_end_10m(series: pd.Series) -> pd.Timestamp:
    timestamps = _phase2_time_basis(series)
    return timestamps.max().floor("10min")


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _load_cpu_metrics(path: Path, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    cpu = pd.read_parquet(
        path,
        columns=[
            "node_id",
            "timestamp",
            "cpu1_utilization",
            "cpu2_utilization",
            "cpu_utilization_mean",
            "cpu1_frequency",
            "cpu2_frequency",
            "cpu1_temp",
            "cpu2_temp",
        ],
    )
    cpu["interval_start"] = _floor_to_10m(cpu["timestamp"])
    cpu = cpu.loc[
        (cpu["interval_start"] >= window_start) & (cpu["interval_start"] <= window_end)
    ].copy()
    cpu_grouped = (
        cpu.groupby(["node_id", "interval_start"], as_index=False)
        .agg(
            cpu1_utilization_mean=("cpu1_utilization", "mean"),
            cpu2_utilization_mean=("cpu2_utilization", "mean"),
            cpu_utilization_mean=("cpu_utilization_mean", "mean"),
            cpu1_frequency_mean=("cpu1_frequency", "mean"),
            cpu2_frequency_mean=("cpu2_frequency", "mean"),
            cpu1_temp_mean=("cpu1_temp", "mean"),
            cpu2_temp_mean=("cpu2_temp", "mean"),
            cpu_samples=("timestamp", "count"),
        )
    )
    return cpu_grouped


def _load_gpu_metrics(path: Path, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    gpu = pd.read_parquet(
        path,
        columns=[
            "node_id",
            "timestamp",
            "gpu_index",
            "is_healthy",
            "power_consum",
            "utilization",
            "memory",
            "temperature",
        ],
    )
    gpu["interval_start"] = _floor_to_10m(gpu["timestamp"])
    gpu = gpu.loc[
        (gpu["interval_start"] >= window_start) & (gpu["interval_start"] <= window_end)
    ].copy()
    gpu["is_healthy_numeric"] = gpu["is_healthy"].astype("float64")
    gpu_grouped = (
        gpu.groupby(["node_id", "interval_start"], as_index=False)
        .agg(
            gpu_utilization_mean=("utilization", "mean"),
            gpu_memory_mean=("memory", "mean"),
            gpu_temperature_mean=("temperature", "mean"),
            gpu_power_mean=("power_consum", "mean"),
            gpu_healthy_ratio=("is_healthy_numeric", "mean"),
            gpu_cards_seen=("gpu_index", "nunique"),
            gpu_samples=("gpu_index", "count"),
        )
    )
    return gpu_grouped


def _load_memory_metrics(path: Path, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    memory = pd.read_parquet(
        path,
        columns=[
            "node_id",
            "timestamp",
            "memory_used_percent",
            "memory_used_mb",
            "swap_used_percent",
            "page_faults",
            "memory_bandwidth",
        ],
    )
    memory["interval_start"] = _floor_to_10m(memory["timestamp"])
    memory = memory.loc[
        (memory["interval_start"] >= window_start) & (memory["interval_start"] <= window_end)
    ].copy()
    memory_grouped = (
        memory.groupby(["node_id", "interval_start"], as_index=False)
        .agg(
            memory_used_percent_mean=("memory_used_percent", "mean"),
            memory_used_mb_mean=("memory_used_mb", "mean"),
            swap_used_percent_mean=("swap_used_percent", "mean"),
            page_faults_mean=("page_faults", "mean"),
            memory_bandwidth_mean=("memory_bandwidth", "mean"),
            memory_samples=("timestamp", "count"),
        )
    )
    return memory_grouped


def _parse_node_ids(series: pd.Series) -> pd.Series:
    return series.apply(
        lambda value: json.loads(value)
        if isinstance(value, str) and value.strip()
        else []
    )


def _compute_overlap_window(
    cpu_path: Path,
    gpu_path: Path,
    memory_path: Path,
    job_path: Path,
) -> tuple[pd.Timestamp, pd.Timestamp, dict[str, dict[str, str]]]:
    cpu_ts = pd.read_parquet(cpu_path, columns=["timestamp"])["timestamp"]
    gpu_ts = pd.read_parquet(gpu_path, columns=["timestamp"])["timestamp"]
    memory_ts = pd.read_parquet(memory_path, columns=["timestamp"])["timestamp"]
    job_ts = pd.read_parquet(job_path, columns=["job_earliest_ts", "job_latest_ts"])

    starts = {
        "cpu": _window_start_10m(cpu_ts),
        "gpu": _window_start_10m(gpu_ts),
        "memory": _window_start_10m(memory_ts),
        "jobs": _window_start_10m(job_ts["job_earliest_ts"]),
    }
    ends = {
        "cpu": _window_end_10m(cpu_ts),
        "gpu": _window_end_10m(gpu_ts),
        "memory": _window_end_10m(memory_ts),
        "jobs": _window_end_10m(job_ts["job_latest_ts"]),
    }

    window_start = max(starts.values())
    window_end = min(ends.values())
    if window_start > window_end:
        raise ValueError("No overlapping time window exists across CPU, GPU, memory, and job data.")

    coverage = {
        name: {
            "start": starts[name].isoformat(),
            "end": ends[name].isoformat(),
        }
        for name in starts
    }
    return window_start, window_end, coverage


def _build_dense_interval_grid(
    cpu_grouped: pd.DataFrame,
    gpu_grouped: pd.DataFrame,
    memory_grouped: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    node_ids = pd.Index(
        sorted(
            set(cpu_grouped["node_id"].dropna().astype(str))
            | set(gpu_grouped["node_id"].dropna().astype(str))
            | set(memory_grouped["node_id"].dropna().astype(str))
        ),
        dtype="string",
    )
    interval_index = pd.date_range(window_start, window_end, freq="10min")
    grid_index = pd.MultiIndex.from_product(
        [node_ids, interval_index],
        names=["node_id", "interval_start"],
    )
    return grid_index.to_frame(index=False)


def _continuity_stats(
    grouped: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> dict[str, int]:
    if grouped.empty:
        return {
            "nodes_with_data": 0,
            "expected_intervals_per_node": 0,
            "missing_intervals_before_densification": 0,
            "largest_gap_intervals_before_densification": 0,
        }

    expected_intervals = len(pd.date_range(window_start, window_end, freq="10min"))
    missing_total = 0
    largest_gap_intervals = 0

    for _, node_slice in grouped.groupby("node_id"):
        intervals = (
            pd.to_datetime(node_slice["interval_start"])
            .sort_values()
            .drop_duplicates()
            .reset_index(drop=True)
        )
        missing_total += expected_intervals - len(intervals)
        diffs = intervals.diff().dropna()
        if diffs.empty:
            continue
        max_gap = int(diffs.max() / pd.Timedelta(minutes=10))
        if max_gap > largest_gap_intervals:
            largest_gap_intervals = max_gap

    return {
        "nodes_with_data": int(grouped["node_id"].nunique()),
        "expected_intervals_per_node": expected_intervals,
        "missing_intervals_before_densification": int(missing_total),
        "largest_gap_intervals_before_densification": int(max(largest_gap_intervals - 1, 0)),
    }


def _explode_jobs(
    path: Path,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, int]]:
    jobs = pd.read_parquet(
        path,
        columns=[
            "job_id",
            "queue",
            "job_state",
            "primary_node_id",
            "node_ids_json",
            "node_count",
            "qtime_ts",
            "stime_ts",
            "etime_ts",
            "mtime_ts",
            "job_earliest_ts",
            "job_latest_ts",
            "Resource_List.ncpus",
            "Resource_List.ngpus",
            "Resource_List.nodect",
        ],
    )
    for column in [
        "qtime_ts",
        "stime_ts",
        "etime_ts",
        "mtime_ts",
        "job_earliest_ts",
        "job_latest_ts",
    ]:
        jobs[column] = _phase2_time_basis(jobs[column])
    jobs["node_ids"] = _parse_node_ids(jobs["node_ids_json"])
    jobs["derived_node_count"] = jobs["node_ids"].apply(len).astype("Int64")
    jobs["node_count"] = _safe_numeric(jobs["node_count"]).astype("Int64")
    jobs["node_count_effective"] = jobs["node_count"].fillna(jobs["derived_node_count"]).astype("Int64")
    jobs["allocated_cores_total"] = _safe_numeric(jobs["Resource_List.ncpus"])
    jobs["allocated_gpus_total"] = _safe_numeric(jobs["Resource_List.ngpus"])
    jobs["requested_nodes_total"] = _safe_numeric(jobs["Resource_List.nodect"])
    jobs["node_ids_effective"] = jobs["node_ids"].where(jobs["derived_node_count"] > 0, jobs.apply(lambda row: [row["primary_node_id"]] if pd.notna(row["primary_node_id"]) else [], axis=1))

    stats = {
        "jobs_loaded": int(len(jobs)),
        "jobs_without_nodes": int((jobs["node_ids_effective"].apply(len) == 0).sum()),
    }

    mapped = jobs.loc[jobs["node_ids_effective"].apply(len) > 0].copy()
    mapped = mapped.explode("node_ids_effective").rename(columns={"node_ids_effective": "node_id"})
    mapped["node_id"] = mapped["node_id"].astype("string")
    divisor = mapped["node_count_effective"].replace({0: pd.NA}).astype("float64")
    mapped["allocated_cores_per_node"] = mapped["allocated_cores_total"] / divisor
    mapped["allocated_gpus_per_node"] = mapped["allocated_gpus_total"] / divisor
    mapped["job_start_ts"] = (
        mapped["stime_ts"]
        .fillna(mapped["qtime_ts"])
        .fillna(mapped["job_earliest_ts"])
    )
    mapped["job_end_ts"] = mapped[
        ["etime_ts", "mtime_ts", "job_latest_ts", "job_start_ts"]
    ].max(axis=1)
    mapped = mapped.loc[
        (mapped["job_end_ts"] >= window_start) & (mapped["job_start_ts"] <= window_end)
    ].copy()
    mapped["job_start_ts"] = mapped["job_start_ts"].clip(lower=window_start, upper=window_end)
    mapped["job_end_ts"] = mapped["job_end_ts"].clip(lower=window_start, upper=window_end)

    mapped = mapped[
        [
            "job_id",
            "node_id",
            "queue",
            "job_state",
            "qtime_ts",
            "stime_ts",
            "etime_ts",
            "mtime_ts",
            "job_start_ts",
            "job_end_ts",
            "job_earliest_ts",
            "job_latest_ts",
            "node_count_effective",
            "requested_nodes_total",
            "allocated_cores_total",
            "allocated_gpus_total",
            "allocated_cores_per_node",
            "allocated_gpus_per_node",
        ]
    ].sort_values(["job_id", "node_id"]).reset_index(drop=True)

    stats["jobs_overlapping_window"] = int(mapped["job_id"].nunique())
    stats["mapped_rows_written"] = int(len(mapped))
    return mapped, stats


def run_phase_02(config: dict) -> dict[str, object]:
    processed_dir = resolve_repo_path(config, config["processed_data_dir"])

    cpu_path = processed_dir / "cpu_raw.parquet"
    gpu_path = processed_dir / "gpu_raw.parquet"
    memory_path = processed_dir / "memory_raw.parquet"
    job_path = processed_dir / "job_raw.parquet"

    node_metrics_output = processed_dir / "node_metrics.parquet"
    job_mapped_output = processed_dir / "job_mapped.parquet"
    summary_output = processed_dir / "phase_02_summary.json"

    window_start, window_end, source_window_coverage = _compute_overlap_window(
        cpu_path,
        gpu_path,
        memory_path,
        job_path,
    )

    cpu_grouped = _load_cpu_metrics(cpu_path, window_start, window_end)
    gpu_grouped = _load_gpu_metrics(gpu_path, window_start, window_end)
    memory_grouped = _load_memory_metrics(memory_path, window_start, window_end)
    continuity_before_densification = _continuity_stats(
        pd.concat(
            [
                cpu_grouped[["node_id", "interval_start"]],
                gpu_grouped[["node_id", "interval_start"]],
                memory_grouped[["node_id", "interval_start"]],
            ],
            ignore_index=True,
        ).drop_duplicates(),
        window_start,
        window_end,
    )

    node_metrics = cpu_grouped.merge(gpu_grouped, on=["node_id", "interval_start"], how="outer")
    node_metrics = node_metrics.merge(memory_grouped, on=["node_id", "interval_start"], how="outer")
    dense_grid = _build_dense_interval_grid(
        cpu_grouped,
        gpu_grouped,
        memory_grouped,
        window_start,
        window_end,
    )
    node_metrics = dense_grid.merge(node_metrics, on=["node_id", "interval_start"], how="left")
    node_metrics["telemetry_present_any"] = node_metrics[
        ["cpu_samples", "gpu_samples", "memory_samples"]
    ].notna().any(axis=1)
    node_metrics = node_metrics.sort_values(["interval_start", "node_id"]).reset_index(drop=True)
    node_metrics.to_parquet(node_metrics_output, index=False)

    job_mapped, job_stats = _explode_jobs(job_path, window_start, window_end)
    job_mapped.to_parquet(job_mapped_output, index=False)

    summary = {
        "seed": config["seed"],
        "assumptions": {
            "interval_alignment": "all telemetry timestamps are floored to 10-minute boundaries within the common overlap window",
            "cpu_aggregation": "mean over 5-minute samples within each 10-minute node interval",
            "gpu_aggregation": "mean across GPU cards at node level within each 10-minute interval",
            "memory_aggregation": "mean within each 10-minute node interval",
            "job_mapping": "one row per job-node mapping; per-node allocations are total requested resources divided by effective node count; invalid end timestamps are repaired by taking the latest available job timestamp",
            "timezone_handling": "all sources are normalized onto a shared UTC-based timeline before 10-minute alignment",
            "continuity_handling": "node_metrics is densified to a full node-by-interval grid across the common overlap window; telemetry gaps remain explicit as missing values",
        },
        "common_overlap_window": {
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
            "duration_days": (window_end - window_start).total_seconds() / 86400,
            "source_window_coverage": source_window_coverage,
        },
        "outputs": {
            "node_metrics_parquet": {
                "path": str(node_metrics_output),
                "rows_written": int(len(node_metrics)),
                "columns": node_metrics.columns.tolist(),
                "coverage_start": None if node_metrics.empty else node_metrics["interval_start"].min().isoformat(),
                "coverage_end": None if node_metrics.empty else node_metrics["interval_start"].max().isoformat(),
            },
            "job_mapped_parquet": {
                "path": str(job_mapped_output),
                "rows_written": int(len(job_mapped)),
                "columns": job_mapped.columns.tolist(),
            },
        },
        "source_shapes": {
            "cpu_10m_rows": int(len(cpu_grouped)),
            "gpu_10m_rows": int(len(gpu_grouped)),
            "memory_10m_rows": int(len(memory_grouped)),
        },
        "continuity": continuity_before_densification,
        "job_mapping_stats": job_stats,
    }

    with summary_output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return summary
