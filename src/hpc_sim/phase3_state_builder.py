from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


INTERVAL = pd.Timedelta(minutes=10)


def _timestamp_ns(series: pd.Series) -> np.ndarray:
    timestamps = pd.to_datetime(series, errors="coerce")
    if hasattr(timestamps.dt, "tz") and timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert("UTC").dt.tz_localize(None)
    return timestamps.astype("datetime64[ns]").to_numpy().astype("int64")


def _normalize_percent_util(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    finite = values.dropna()
    if finite.empty:
        return values
    if float(finite.quantile(0.99)) > 1.5:
        values = values / 100.0
    return values.clip(lower=0.0, upper=1.0)


def _load_phase_02_payload(phase_02_dir: Path) -> dict:
    payload_path = phase_02_dir / "payload.json"
    with payload_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_phase_02_payload(phase_02_dir: Path) -> dict:
    candidate_dirs = [phase_02_dir]
    candidate_dirs.extend(sorted(phase_02_dir.parent.parent.glob("*/phase_02_preprocessing")))
    for candidate_dir in candidate_dirs:
        payload_path = candidate_dir / "payload.json"
        if not payload_path.exists():
            continue
        with payload_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("processed_outputs", {}).get("node_metrics_parquet", {}).get("path"):
            return payload
    raise FileNotFoundError("Unable to locate a valid phase_02_preprocessing payload")


def _expand_active_job_intervals(job_mapped: pd.DataFrame) -> pd.DataFrame:
    jobs = job_mapped.copy()
    jobs["job_start_interval"] = pd.to_datetime(jobs["job_start_ts"]).dt.floor("10min")
    jobs["job_end_interval"] = pd.to_datetime(jobs["job_end_ts"]).dt.floor("10min")

    durations = (
        (jobs["job_end_interval"] - jobs["job_start_interval"]) / INTERVAL
    ).astype("int64") + 1
    jobs = jobs.loc[durations > 0].reset_index(drop=True)
    durations = durations.loc[durations > 0].reset_index(drop=True)
    if jobs.empty:
        return pd.DataFrame(
            columns=[
                "node_id",
                "interval_start",
                "job_id",
                "allocated_cores_per_node",
                "allocated_gpus_per_node",
            ]
        )

    repeat_index = np.repeat(np.arange(len(jobs)), durations.to_numpy())
    group_offsets = np.repeat(np.cumsum(durations.to_numpy()) - durations.to_numpy(), durations.to_numpy())
    interval_offsets = np.arange(len(repeat_index), dtype=np.int64) - group_offsets
    interval_start_ns = _timestamp_ns(jobs["job_start_interval"])
    expanded_interval_ns = np.repeat(interval_start_ns, durations.to_numpy()) + (
        interval_offsets * INTERVAL.value
    )

    expanded = pd.DataFrame(
        {
            "node_id": jobs.loc[repeat_index, "node_id"].to_numpy(),
            "interval_start": pd.to_datetime(expanded_interval_ns),
            "job_id": jobs.loc[repeat_index, "job_id"].to_numpy(),
            "allocated_cores_per_node": jobs.loc[repeat_index, "allocated_cores_per_node"].to_numpy(),
            "allocated_gpus_per_node": jobs.loc[repeat_index, "allocated_gpus_per_node"].to_numpy(),
        }
    )
    return expanded


def _aggregate_active_jobs(expanded_jobs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if expanded_jobs.empty:
        empty = pd.DataFrame(
            columns=[
                "node_id",
                "interval_start",
                "allocated_cores",
                "allocated_gpus",
                "job_id",
                "active_job_count",
                "job_ids_json",
            ]
        )
        stats = {
            "expanded_active_rows": 0,
            "node_interval_slots_with_jobs": 0,
            "multi_job_slots": 0,
            "max_jobs_per_slot": 0,
        }
        return empty, stats

    representative = (
        expanded_jobs.sort_values(
            ["node_id", "interval_start", "allocated_cores_per_node", "allocated_gpus_per_node", "job_id"],
            ascending=[True, True, False, False, True],
        )
        .groupby(["node_id", "interval_start"], as_index=False)
        .first()[["node_id", "interval_start", "job_id"]]
    )

    aggregated = (
        expanded_jobs.groupby(["node_id", "interval_start"], as_index=False)
        .agg(
            allocated_cores=("allocated_cores_per_node", "sum"),
            allocated_gpus=("allocated_gpus_per_node", "sum"),
            active_job_count=("job_id", "nunique"),
            job_ids_json=("job_id", lambda values: json.dumps(sorted(pd.unique(values).tolist()))),
        )
    )
    aggregated = aggregated.merge(representative, on=["node_id", "interval_start"], how="left")
    aggregated = aggregated[
        [
            "node_id",
            "interval_start",
            "allocated_cores",
            "allocated_gpus",
            "job_id",
            "active_job_count",
            "job_ids_json",
        ]
    ]

    stats = {
        "expanded_active_rows": int(len(expanded_jobs)),
        "node_interval_slots_with_jobs": int(len(aggregated)),
        "multi_job_slots": int((aggregated["active_job_count"] > 1).sum()),
        "max_jobs_per_slot": int(aggregated["active_job_count"].max()),
    }
    return aggregated, stats


def run_phase_03(input_dir: Path, output_dir: Path) -> dict[str, object]:
    phase_02_payload = _resolve_phase_02_payload(input_dir)
    node_metrics_path = Path(phase_02_payload["processed_outputs"]["node_metrics_parquet"]["path"])
    job_mapped_path = Path(phase_02_payload["processed_outputs"]["job_mapped_parquet"]["path"])
    print("phase3 loading node_metrics", flush=True)

    node_metrics = pd.read_parquet(
        node_metrics_path,
        columns=[
            "node_id",
            "interval_start",
            "cpu_utilization_mean",
            "gpu_utilization_mean",
            "memory_used_percent_mean",
            "telemetry_present_any",
        ],
    )
    node_metrics["interval_start"] = pd.to_datetime(node_metrics["interval_start"])
    print(f"phase3 loaded node_metrics rows={len(node_metrics)}", flush=True)

    print("phase3 loading job_mapped", flush=True)
    job_mapped = pd.read_parquet(
        job_mapped_path,
        columns=[
            "job_id",
            "node_id",
            "job_start_ts",
            "job_end_ts",
            "allocated_cores_per_node",
            "allocated_gpus_per_node",
        ],
    )
    job_mapped["job_start_ts"] = pd.to_datetime(job_mapped["job_start_ts"])
    job_mapped["job_end_ts"] = pd.to_datetime(job_mapped["job_end_ts"])
    print(f"phase3 loaded job_mapped rows={len(job_mapped)}", flush=True)

    jobs_by_node = {
        node_id: frame.reset_index(drop=True)
        for node_id, frame in job_mapped.groupby("node_id", sort=False)
    }
    print(f"phase3 grouped jobs_by_node nodes={len(jobs_by_node)}", flush=True)

    output_path = output_dir / "state_cache.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    active_job_stats = {
        "expanded_active_rows": 0,
        "node_interval_slots_with_jobs": 0,
        "multi_job_slots": 0,
        "max_jobs_per_slot": 0,
    }

    writer: pq.ParquetWriter | None = None
    rows_written = 0
    timestamp_count = 0
    node_count = 0
    explicit_missing_telemetry_rows = 0
    metrics_missing_after_ffill = {
        "cpu_util_missing": 0,
        "gpu_util_missing": 0,
        "memory_util_missing": 0,
    }

    try:
        for node_idx, (node_id, node_state) in enumerate(node_metrics.groupby("node_id", sort=True), start=1):
            if node_idx == 1:
                print(f"phase3 entering node loop first_node={node_id}", flush=True)
            node_state = node_state.sort_values("interval_start").copy()
            node_state[[
                "cpu_utilization_mean",
                "gpu_utilization_mean",
                "memory_used_percent_mean",
            ]] = node_state[[
                "cpu_utilization_mean",
                "gpu_utilization_mean",
                "memory_used_percent_mean",
            ]].ffill()
            node_state["gpu_utilization_mean"] = node_state["gpu_utilization_mean"].fillna(0.0)

            raw_node_jobs = jobs_by_node.get(str(node_id))
            if raw_node_jobs is None:
                node_state["allocated_cores"] = 0.0
                node_state["allocated_gpus"] = 0.0
                node_state["job_id"] = pd.Series([pd.NA] * len(node_state), dtype="string")
                node_state["active_job_count"] = 0
            else:
                expanded_jobs = _expand_active_job_intervals(raw_node_jobs)
                node_jobs, node_job_stats = _aggregate_active_jobs(expanded_jobs)
                active_job_stats["expanded_active_rows"] += node_job_stats["expanded_active_rows"]
                active_job_stats["node_interval_slots_with_jobs"] += node_job_stats["node_interval_slots_with_jobs"]
                active_job_stats["multi_job_slots"] += node_job_stats["multi_job_slots"]
                active_job_stats["max_jobs_per_slot"] = max(
                    active_job_stats["max_jobs_per_slot"],
                    node_job_stats["max_jobs_per_slot"],
                )
                node_jobs = node_jobs.drop(columns=["node_id"]).reset_index(drop=True)
                node_state = node_state.merge(node_jobs, on="interval_start", how="left")
                node_state["allocated_cores"] = node_state["allocated_cores"].fillna(0.0)
                node_state["allocated_gpus"] = node_state["allocated_gpus"].fillna(0.0)
                node_state["job_id"] = node_state["job_id"].astype("string")
                node_state["active_job_count"] = node_state["active_job_count"].fillna(0).astype("int64")

            node_state = node_state.rename(
                columns={
                    "interval_start": "timestamp",
                    "cpu_utilization_mean": "cpu_util",
                    "gpu_utilization_mean": "gpu_util",
                    "memory_used_percent_mean": "memory_util",
                }
            )
            node_state = node_state[
                [
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
                ]
            ]
            node_state["cpu_util"] = _normalize_percent_util(node_state["cpu_util"])
            node_state["gpu_util"] = _normalize_percent_util(node_state["gpu_util"])
            node_state["memory_util"] = _normalize_percent_util(node_state["memory_util"])

            rows_written += len(node_state)
            timestamp_count = max(timestamp_count, int(node_state["timestamp"].nunique()))
            node_count += 1
            explicit_missing_telemetry_rows += int((~node_state["telemetry_present_any"]).sum())
            metrics_missing_after_ffill["cpu_util_missing"] += int(node_state["cpu_util"].isna().sum())
            metrics_missing_after_ffill["gpu_util_missing"] += int(node_state["gpu_util"].isna().sum())
            metrics_missing_after_ffill["memory_util_missing"] += int(node_state["memory_util"].isna().sum())

            table = pa.Table.from_pandas(node_state, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
            if node_idx % 50 == 0:
                print(
                    f"phase3 progress nodes={node_idx} rows_written={rows_written}",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()

    state_sorted = pd.read_parquet(output_path).sort_values(["timestamp", "node_id"]).reset_index(drop=True)
    state_sorted.to_parquet(output_path, index=False)

    return {
        "status": "complete",
        "state_cache_path": str(output_path),
        "rows_written": int(rows_written),
        "columns": [
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
        "coverage_start": None if node_metrics.empty else pd.to_datetime(node_metrics["interval_start"]).min().isoformat(),
        "coverage_end": None if node_metrics.empty else pd.to_datetime(node_metrics["interval_start"]).max().isoformat(),
        "node_count": int(node_count),
        "timestamp_count": int(timestamp_count),
        "explicit_missing_telemetry_rows": int(explicit_missing_telemetry_rows),
        "metrics_missing_after_ffill": metrics_missing_after_ffill,
        "active_job_stats": active_job_stats,
        "multi_job_resolution_rule": (
            "allocated resources are summed across active jobs; job_id is the dominant active job "
            "ordered by allocated_cores_per_node desc, allocated_gpus_per_node desc, job_id asc"
        ),
        "utilization_scale": "cpu_util, gpu_util, and memory_util are normalized to fractions in [0, 1]",
    }
