from __future__ import annotations

import json
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import resolve_repo_path
from .phase5_reclamation import _gpu_reclaimable_cards


CPU_CORES_PER_NODE = 128.0
INTERVAL_SECONDS = 600
SCENARIO_RE = re.compile(r"^(?P<mode>[a-z]+)_threshold_(?P<threshold>\d+\.\d+)$")
CHECKPOINT_DIRNAME = "_resume"
CHECKPOINT_STATE_FILENAME = "checkpoint_state.json"


@dataclass
class QueueJob:
    job_id: str
    cpu_required: float
    gpu_required: float
    remaining_seconds: int
    initial_remaining_seconds: int
    qtime_ts: pd.Timestamp
    queue: str


def _load_phase_payload(phase_dir: Path) -> dict:
    payload_path = phase_dir / "payload.json"
    with payload_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_phase_03_payload(scenario_dir: Path, results_root: Path) -> dict:
    candidate_dirs = [scenario_dir / "phase_03_state_construction"]
    candidate_dirs.extend(sorted(results_root.glob("*/phase_03_state_construction")))
    for candidate_dir in candidate_dirs:
        payload_path = candidate_dir / "payload.json"
        if not payload_path.exists():
            continue
        with payload_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("state_cache_path"):
            return payload
    raise FileNotFoundError("Unable to locate a valid phase_03_state_construction payload")


def _parse_scenario(scenario: str) -> tuple[str, float]:
    match = SCENARIO_RE.match(scenario)
    if not match:
        raise ValueError(f"Unable to parse scenario: {scenario}")
    return match.group("mode"), float(match.group("threshold"))


def _time_basis(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce")
    if hasattr(timestamps.dt, "tz") and timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert("UTC").dt.tz_localize(None)
    return timestamps


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalize_util(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric.dropna()
    if finite.empty:
        return numeric
    if float(finite.quantile(0.99)) > 1.5:
        numeric = numeric / 100.0
    return numeric.clip(lower=0.0, upper=1.0)


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


def _primary_reserve_util(config: dict, mode: str, threshold: float) -> float:
    if mode == "aggressive":
        return threshold
    if mode == "buffered":
        return float(config["buffers"][f"{threshold:.2f}"])
    return threshold


def _reference_threshold_scenario(threshold: float) -> str:
    return f"original_threshold_{threshold:.2f}"


def _load_queue_catalog(config: dict, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
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
    jobs["etime_ts"] = _time_basis(jobs["etime_ts"])
    jobs["cpu_required"] = _safe_numeric(jobs["Resource_List.ncpus"]).fillna(0.0)
    jobs["gpu_required"] = _safe_numeric(jobs["Resource_List.ngpus"]).fillna(0.0)
    jobs["remaining_seconds"] = _resolve_runtime_seconds(jobs)
    jobs["queue"] = jobs["queue"].astype("string")
    jobs["job_id"] = jobs["job_id"].astype("string")
    jobs = jobs.loc[
        jobs["qtime_ts"].notna()
        & (jobs["qtime_ts"] >= window_start)
        & (jobs["qtime_ts"] <= window_end)
        & ((jobs["cpu_required"] > 0) | (jobs["gpu_required"] > 0))
        & (jobs["remaining_seconds"] > 0)
    ].copy()
    jobs = jobs.sort_values(["qtime_ts", "job_id"]).reset_index(drop=True)
    return jobs[
        [
            "job_id",
            "queue",
            "qtime_ts",
            "cpu_required",
            "gpu_required",
            "remaining_seconds",
        ]
    ]


def _load_candidate_slots(results_root: Path, threshold: float) -> pd.DataFrame:
    scenario_dir = results_root / _reference_threshold_scenario(threshold)
    phase_05_payload = _load_phase_payload(scenario_dir / "phase_05_reclamation")
    reclamation = pd.read_parquet(
        Path(phase_05_payload["reclamation_plan_path"]),
        columns=[
            "timestamp",
            "node_id",
            "cpu_util",
            "gpu_util",
            "memory_util",
            "reclaimable_cores",
            "reclaimable_gpus",
        ],
    )
    reclamation["timestamp"] = pd.to_datetime(reclamation["timestamp"])
    reclamation["cpu_util"] = _normalize_util(reclamation["cpu_util"])
    reclamation["gpu_util"] = _normalize_util(reclamation["gpu_util"])
    reclamation["memory_util"] = _normalize_util(reclamation["memory_util"])
    return reclamation.sort_values(
        ["timestamp", "reclaimable_gpus", "reclaimable_cores", "node_id"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def _load_state_envelope(results_root: Path, threshold: float) -> pd.DataFrame:
    scenario_dir = results_root / _reference_threshold_scenario(threshold)
    phase_03_payload = _resolve_phase_03_payload(scenario_dir, results_root)
    state = pd.read_parquet(
        Path(phase_03_payload["state_cache_path"]),
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
    state["cpu_util"] = _normalize_util(state["cpu_util"])
    state["gpu_util"] = _normalize_util(state["gpu_util"])
    state["memory_util"] = _normalize_util(state["memory_util"])
    state["reclaimable_cores_now"] = (
        CPU_CORES_PER_NODE * (1.0 - state["cpu_util"].fillna(1.0))
    ).clip(lower=0.0)
    state["reclaimable_gpus_now"] = [
        _gpu_reclaimable_cards(allocated_gpus, gpu_util)
        for allocated_gpus, gpu_util in zip(state["allocated_gpus"], state["gpu_util"])
    ]
    return state.sort_values(["timestamp", "node_id"]).reset_index(drop=True)


def _select_best_fit_job(
    feasible_window: list[str],
    jobs_by_id: dict[str, QueueJob],
    reclaimable_cores: float,
    reclaimable_gpus: float,
    excluded_job_ids: set[str],
) -> str | None:
    best_job_id: str | None = None
    best_key: tuple | None = None
    for rank, job_id in enumerate(feasible_window, start=1):
        job = jobs_by_id[job_id]
        cpu_slack = reclaimable_cores - job.cpu_required
        gpu_slack = reclaimable_gpus - job.gpu_required

        candidate_key = (
            gpu_slack,
            cpu_slack,
            job.remaining_seconds,
            job.gpu_required,
            job.cpu_required,
            rank,
            job.job_id,
        )
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best_job_id = job_id
    return best_job_id


def _pooling_potential(
    candidate_slice: pd.DataFrame,
    feasible_window: list[str],
    jobs_by_id: dict[str, QueueJob],
    excluded_job_ids: set[str],
) -> bool:
    if not feasible_window:
        return False
    total_cores = float(candidate_slice["reclaimable_cores"].sum())
    total_gpus = float(candidate_slice["reclaimable_gpus"].sum())
    for job_id in feasible_window:
        job = jobs_by_id[job_id]
        if job.cpu_required <= total_cores and job.gpu_required <= total_gpus:
            return True
    return False


def _feasible_window(
    waiting_queue: deque[str],
    jobs_by_id: dict[str, QueueJob],
    reclaimable_cores: float,
    reclaimable_gpus: float,
    excluded_job_ids: set[str],
    feasible_size: int = 10,
    scan_limit: int = 200,
) -> list[str]:
    feasible: list[str] = []
    scanned = 0
    for job_id in waiting_queue:
        if scanned >= scan_limit or len(feasible) >= feasible_size:
            break
        scanned += 1
        if job_id in excluded_job_ids:
            continue
        job = jobs_by_id[job_id]
        if job.cpu_required > reclaimable_cores or job.gpu_required > reclaimable_gpus:
            continue
        feasible.append(job_id)
    return feasible


def _active_cpu_alloc(job_ids: list[str], jobs_by_id: dict[str, QueueJob]) -> float:
    return float(sum(jobs_by_id[job_id].cpu_required for job_id in job_ids))


def _active_gpu_alloc(job_ids: list[str], jobs_by_id: dict[str, QueueJob]) -> float:
    return float(sum(jobs_by_id[job_id].gpu_required for job_id in job_ids))


def _dominant_job(job_ids: list[str], jobs_by_id: dict[str, QueueJob]) -> str | None:
    if not job_ids:
        return None
    return sorted(
        job_ids,
        key=lambda job_id: (
            -jobs_by_id[job_id].gpu_required,
            -jobs_by_id[job_id].cpu_required,
            jobs_by_id[job_id].remaining_seconds,
            job_id,
        ),
    )[0]


def _preemption_choice(job_ids: list[str], jobs_by_id: dict[str, QueueJob]) -> str:
    return sorted(
        job_ids,
        key=lambda job_id: (
            -jobs_by_id[job_id].gpu_required,
            -jobs_by_id[job_id].cpu_required,
            -jobs_by_id[job_id].remaining_seconds,
            job_id,
        ),
    )[0]


def _remove_job_from_scanned_prefix(waiting_queue: deque[str], selected_job_id: str, scan_limit: int = 200) -> None:
    prefix_size = min(scan_limit, len(waiting_queue))
    prefix = [waiting_queue.popleft() for _ in range(prefix_size)]
    removed = False
    remaining_prefix: list[str] = []
    for job_id in prefix:
        if not removed and job_id == selected_job_id:
            removed = True
            continue
        remaining_prefix.append(job_id)
    for job_id in reversed(remaining_prefix):
        waiting_queue.appendleft(job_id)


def _append_waiting_job(
    waiting_queue: deque[str],
    waiting_job_ids: set[str],
    job_id: str,
) -> None:
    if job_id in waiting_job_ids:
        return
    waiting_queue.append(job_id)
    waiting_job_ids.add(job_id)


def _appendleft_waiting_job(
    waiting_queue: deque[str],
    waiting_job_ids: set[str],
    job_id: str,
) -> None:
    if job_id in waiting_job_ids:
        return
    waiting_queue.appendleft(job_id)
    waiting_job_ids.add(job_id)


def _remove_waiting_job_from_scanned_prefix(
    waiting_queue: deque[str],
    waiting_job_ids: set[str],
    selected_job_id: str,
    scan_limit: int = 200,
) -> None:
    _remove_job_from_scanned_prefix(waiting_queue, selected_job_id, scan_limit=scan_limit)
    waiting_job_ids.discard(selected_job_id)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


def _flush_log_batches(log_batches: list[pd.DataFrame], batch_dir: Path, next_batch_idx: int) -> int:
    if not log_batches:
        return next_batch_idx
    batch_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(log_batches, ignore_index=True).sort_values(["timestamp", "node_id"]).reset_index(drop=True)
    part_path = batch_dir / f"part-{next_batch_idx:05d}.parquet"
    _write_parquet(combined, part_path)
    log_batches.clear()
    return next_batch_idx + 1


def _checkpoint_payload(
    *,
    next_timestamp_index: int,
    queue_pointer: int,
    waiting_queue: deque[str],
    waiting_job_ids: set[str],
    deferred_requeue: list[str],
    active_by_node: dict[str, list[str]],
    active_job_ids: set[str],
    completed_job_ids: set[str],
    unique_jobs_run: set[str],
    injection_count: int,
    preemption_count: int,
    completion_count: int,
    pooling_timestamps_checked: int,
    pooling_improvement_timestamps: int,
    previous_ts: pd.Timestamp | None,
    next_batch_idx: int,
    jobs_by_id: dict[str, QueueJob],
) -> dict[str, object]:
    mutated_remaining_seconds = {
        job_id: job.remaining_seconds
        for job_id, job in jobs_by_id.items()
        if job.remaining_seconds != job.initial_remaining_seconds
    }
    return {
        "next_timestamp_index": next_timestamp_index,
        "queue_pointer": queue_pointer,
        "waiting_queue": list(waiting_queue),
        "waiting_job_ids": sorted(waiting_job_ids),
        "deferred_requeue": list(deferred_requeue),
        "active_by_node": {node_id: list(job_ids) for node_id, job_ids in active_by_node.items()},
        "active_job_ids": sorted(active_job_ids),
        "completed_job_ids": sorted(completed_job_ids),
        "unique_jobs_run": sorted(unique_jobs_run),
        "injection_count": injection_count,
        "preemption_count": preemption_count,
        "completion_count": completion_count,
        "pooling_timestamps_checked": pooling_timestamps_checked,
        "pooling_improvement_timestamps": pooling_improvement_timestamps,
        "previous_ts": previous_ts.isoformat() if previous_ts is not None else None,
        "next_batch_idx": next_batch_idx,
        "mutated_remaining_seconds": mutated_remaining_seconds,
    }


def _write_checkpoint(checkpoint_path: Path, payload: dict[str, object]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_checkpoint(checkpoint_path: Path) -> dict[str, object]:
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _finalize_simulation_output(batch_dir: Path, output_path: Path) -> int:
    part_paths = sorted(batch_dir.glob("part-*.parquet"))
    if not part_paths:
        empty = pd.DataFrame()
        _write_parquet(empty, output_path)
        return 0

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    try:
        for part_path in part_paths:
            table = pq.read_table(part_path)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
            total_rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    return total_rows


def run_phase_09(
    config: dict,
    phase_08_dir: Path,
    scenario_dir: Path,
    output_dir: Path,
    scenario: str,
) -> dict[str, object]:
    del phase_08_dir
    mode, threshold = _parse_scenario(scenario)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "simulation_log.parquet"
    resume_dir = output_dir / CHECKPOINT_DIRNAME
    checkpoint_path = resume_dir / CHECKPOINT_STATE_FILENAME
    batch_dir = resume_dir / "batches"
    force_restart = os.environ.get("HPC_SIM_FORCE") == "1"

    primary_reserve_util = _primary_reserve_util(config, mode, threshold)

    if mode == "original":
        phase_03_payload = _resolve_phase_03_payload(scenario_dir, scenario_dir.parent)
        state_path = Path(phase_03_payload["state_cache_path"])
        state = pd.read_parquet(
            state_path,
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
        state = state.sort_values(["timestamp", "node_id"]).reset_index(drop=True)
        baseline = state.copy()
        baseline["mode"] = mode
        baseline["threshold"] = threshold
        baseline["allowed_total_util"] = primary_reserve_util
        baseline["is_underutilized"] = False
        baseline["reclaimable_cores"] = 0
        baseline["reclaimable_gpus"] = 0
        baseline["opportunistic_job_id"] = pd.Series([pd.NA] * len(baseline), dtype="string")
        baseline["opportunistic_job_count"] = 0
        baseline["opportunistic_job_ids_json"] = "[]"
        baseline["opportunistic_cpu_allocated"] = 0.0
        baseline["opportunistic_gpu_allocated"] = 0.0
        baseline["opportunistic_remaining_seconds"] = pd.Series([pd.NA] * len(baseline), dtype="Float64")
        baseline["injected_this_interval"] = False
        baseline["preempted_this_interval"] = False
        baseline["completed_this_interval"] = False
        baseline["total_util"] = baseline["cpu_util"].fillna(0.0)
        _write_parquet(baseline, output_path)
        summary = {
            "status": "complete",
            "simulation_log_path": str(output_path),
            "rows_written": int(len(baseline)),
            "coverage_start": baseline["timestamp"].min().isoformat(),
            "coverage_end": baseline["timestamp"].max().isoformat(),
            "node_count": int(baseline["node_id"].nunique()),
            "timestamp_count": int(baseline["timestamp"].nunique()),
            "mode": mode,
            "threshold": threshold,
            "allowed_total_util": primary_reserve_util,
            "primary_reserve_util": primary_reserve_util,
            "injection_count": 0,
            "preemption_count": 0,
            "completion_count": 0,
            "unique_opportunistic_jobs_run": 0,
            "pooling_improvement_timestamps": 0,
            "pooling_timestamps_checked": 0,
            "columns": baseline.columns.tolist(),
        }
        summary_path = output_dir / "simulation_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return summary

    queue_settings = config.get("queue", {})
    feasible_size = int(queue_settings.get("selection_window", 25))
    scan_limit = int(queue_settings.get("scan_limit", max(feasible_size * 40, 1000)))
    max_jobs_per_node = int(queue_settings.get("max_injected_jobs_per_node", 4))
    checkpoint_interval_timestamps = int(queue_settings.get("checkpoint_interval_timestamps", 500))

    candidates = _load_candidate_slots(scenario_dir.parent, threshold)
    state_envelope = _load_state_envelope(scenario_dir.parent, threshold)
    queue_catalog = _load_queue_catalog(
        config,
        window_start=candidates["timestamp"].min(),
        window_end=candidates["timestamp"].max(),
    )
    jobs_by_id = {
        str(row.job_id): QueueJob(
            job_id=str(row.job_id),
            cpu_required=float(row.cpu_required),
            gpu_required=float(row.gpu_required),
            remaining_seconds=int(row.remaining_seconds),
            initial_remaining_seconds=int(row.remaining_seconds),
            qtime_ts=pd.Timestamp(row.qtime_ts),
            queue=str(row.queue),
        )
        for row in queue_catalog.itertuples(index=False)
    }
    queue_rows = list(queue_catalog.itertuples(index=False))
    waiting_queue: deque[str] = deque()
    waiting_job_ids: set[str] = set()
    deferred_requeue: list[str] = []
    active_by_node: dict[str, list[str]] = {}
    active_job_ids: set[str] = set()
    completed_job_ids: set[str] = set()
    unique_jobs_run: set[str] = set()

    queue_pointer = 0
    candidate_by_ts = {
        timestamp: frame.reset_index(drop=True)
        for timestamp, frame in candidates.groupby("timestamp", sort=True)
    }
    state_by_ts = {
        timestamp: frame.reset_index(drop=True)
        for timestamp, frame in state_envelope.groupby("timestamp", sort=True)
    }
    log_batches: list[pd.DataFrame] = []
    next_batch_idx = 0

    injection_count = 0
    preemption_count = 0
    completion_count = 0
    pooling_timestamps_checked = 0
    pooling_improvement_timestamps = 0

    timeline = sorted(candidate_by_ts.keys())
    start_timestamp_index = 0
    previous_ts: pd.Timestamp | None = None

    if force_restart:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if output_path.exists():
            output_path.unlink()
        if batch_dir.exists():
            for stale_part in batch_dir.glob("part-*.parquet"):
                stale_part.unlink()
    elif checkpoint_path.exists():
        checkpoint = _load_checkpoint(checkpoint_path)
        start_timestamp_index = int(checkpoint.get("next_timestamp_index", 0))
        queue_pointer = int(checkpoint.get("queue_pointer", 0))
        waiting_queue = deque(str(job_id) for job_id in checkpoint.get("waiting_queue", []))
        waiting_job_ids = {str(job_id) for job_id in checkpoint.get("waiting_job_ids", [])}
        deferred_requeue = [str(job_id) for job_id in checkpoint.get("deferred_requeue", [])]
        active_by_node = {
            str(node_id): [str(job_id) for job_id in job_ids]
            for node_id, job_ids in checkpoint.get("active_by_node", {}).items()
        }
        active_job_ids = {str(job_id) for job_id in checkpoint.get("active_job_ids", [])}
        completed_job_ids = {str(job_id) for job_id in checkpoint.get("completed_job_ids", [])}
        unique_jobs_run = {str(job_id) for job_id in checkpoint.get("unique_jobs_run", [])}
        injection_count = int(checkpoint.get("injection_count", 0))
        preemption_count = int(checkpoint.get("preemption_count", 0))
        completion_count = int(checkpoint.get("completion_count", 0))
        pooling_timestamps_checked = int(checkpoint.get("pooling_timestamps_checked", 0))
        pooling_improvement_timestamps = int(checkpoint.get("pooling_improvement_timestamps", 0))
        previous_ts_raw = checkpoint.get("previous_ts")
        previous_ts = pd.Timestamp(previous_ts_raw) if previous_ts_raw else None
        next_batch_idx = int(checkpoint.get("next_batch_idx", 0))
        for job_id, remaining_seconds in checkpoint.get("mutated_remaining_seconds", {}).items():
            if job_id in jobs_by_id:
                jobs_by_id[job_id].remaining_seconds = int(remaining_seconds)
        print(
            (
                f"phase9 resume scenario={scenario} start_timestamp_index={start_timestamp_index} "
                f"queue_pointer={queue_pointer} waiting={len(waiting_queue)} "
                f"active_nodes={len(active_by_node)} injections={injection_count} "
                f"preemptions={preemption_count} completions={completion_count}"
            ),
            flush=True,
        )
    else:
        if output_path.exists():
            output_path.unlink()
        if batch_dir.exists():
            for stale_part in batch_dir.glob("part-*.parquet"):
                stale_part.unlink()

    for ts_idx, current_ts in enumerate(timeline[start_timestamp_index:], start=start_timestamp_index + 1):
        if previous_ts is not None and (current_ts - previous_ts) > pd.Timedelta(minutes=10):
            for node_jobs in active_by_node.values():
                deferred_requeue.extend(node_jobs)
            active_by_node.clear()
            active_job_ids.clear()
        previous_ts = current_ts

        if deferred_requeue:
            for job_id in reversed(deferred_requeue):
                if job_id not in active_job_ids and job_id not in completed_job_ids:
                    _appendleft_waiting_job(waiting_queue, waiting_job_ids, job_id)
            deferred_requeue = []

        while queue_pointer < len(queue_rows) and pd.Timestamp(queue_rows[queue_pointer].qtime_ts) <= current_ts:
            job_id = str(queue_rows[queue_pointer].job_id)
            if job_id not in active_job_ids and job_id not in completed_job_ids:
                _append_waiting_job(waiting_queue, waiting_job_ids, job_id)
            queue_pointer += 1

        candidate_slice = candidate_by_ts.get(current_ts)
        state_slice = state_by_ts.get(current_ts)
        candidate_lookup = (
            candidate_slice.set_index("node_id", drop=False)
            if candidate_slice is not None and not candidate_slice.empty
            else None
        )
        state_lookup = (
            state_slice.set_index("node_id", drop=False)
            if state_slice is not None and not state_slice.empty
            else None
        )
        node_events: dict[str, dict[str, bool]] = defaultdict(
            lambda: {
                "injected_this_interval": False,
                "preempted_this_interval": False,
                "completed_this_interval": False,
            }
        )
        excluded_job_ids: set[str] = set()
        deferred_preempted: list[str] = []

        for node_id in list(active_by_node.keys()):
            active_jobs = list(active_by_node.get(node_id, []))
            if state_lookup is None or node_id not in state_lookup.index:
                for job_id in active_jobs:
                    active_job_ids.discard(job_id)
                    excluded_job_ids.add(job_id)
                    deferred_preempted.append(job_id)
                    preemption_count += 1
                node_events[node_id]["preempted_this_interval"] = bool(active_jobs)
                active_by_node.pop(node_id, None)
                continue

            row = state_lookup.loc[node_id]
            for job_id in list(active_jobs):
                job = jobs_by_id[job_id]
                job.remaining_seconds = max(job.remaining_seconds - INTERVAL_SECONDS, 0)
                unique_jobs_run.add(job_id)
                if job.remaining_seconds == 0:
                    completion_count += 1
                    node_events[node_id]["completed_this_interval"] = True
                    active_jobs.remove(job_id)
                    active_job_ids.discard(job_id)
                    completed_job_ids.add(job_id)

            if float(row.cpu_util) > primary_reserve_util:
                while active_jobs:
                    job_id = _preemption_choice(active_jobs, jobs_by_id)
                    active_jobs.remove(job_id)
                    active_job_ids.discard(job_id)
                    excluded_job_ids.add(job_id)
                    deferred_preempted.append(job_id)
                    preemption_count += 1
                    node_events[node_id]["preempted_this_interval"] = True

            while active_jobs and (
                (_active_cpu_alloc(active_jobs, jobs_by_id) > float(row.reclaimable_cores_now))
                or (_active_gpu_alloc(active_jobs, jobs_by_id) > float(row.reclaimable_gpus_now))
            ):
                job_id = _preemption_choice(active_jobs, jobs_by_id)
                active_jobs.remove(job_id)
                active_job_ids.discard(job_id)
                excluded_job_ids.add(job_id)
                deferred_preempted.append(job_id)
                preemption_count += 1
                node_events[node_id]["preempted_this_interval"] = True

            if active_jobs:
                active_by_node[node_id] = active_jobs
            else:
                active_by_node.pop(node_id, None)

        if candidate_slice is not None and not candidate_slice.empty:
            had_no_single_fit = False
            if not candidate_slice.empty:
                for candidate in candidate_slice.itertuples(index=False):
                    node_id = str(candidate.node_id)
                    current_jobs = active_by_node.get(node_id, [])
                    available_cpu = float(candidate.reclaimable_cores) - _active_cpu_alloc(current_jobs, jobs_by_id)
                    available_gpu = float(candidate.reclaimable_gpus) - _active_gpu_alloc(current_jobs, jobs_by_id)
                    feasible_window = _feasible_window(
                        waiting_queue=waiting_queue,
                        jobs_by_id=jobs_by_id,
                        reclaimable_cores=max(available_cpu, 0.0),
                        reclaimable_gpus=max(available_gpu, 0.0),
                        excluded_job_ids=excluded_job_ids,
                        feasible_size=feasible_size,
                        scan_limit=scan_limit,
                    )
                    if not feasible_window:
                        had_no_single_fit = True
                        break

            if had_no_single_fit:
                pooling_timestamps_checked += 1
                if _pooling_potential(
                    candidate_slice=candidate_slice,
                    feasible_window=_feasible_window(
                        waiting_queue=waiting_queue,
                        jobs_by_id=jobs_by_id,
                        reclaimable_cores=float(candidate_slice["reclaimable_cores"].sum()),
                        reclaimable_gpus=float(candidate_slice["reclaimable_gpus"].sum()),
                        excluded_job_ids=excluded_job_ids,
                        feasible_size=feasible_size,
                        scan_limit=scan_limit,
                    ),
                    jobs_by_id=jobs_by_id,
                    excluded_job_ids=excluded_job_ids,
                ):
                    pooling_improvement_timestamps += 1

            for candidate in candidate_slice.itertuples(index=False):
                node_id = str(candidate.node_id)
                active_jobs = list(active_by_node.get(node_id, []))
                injections_this_node = 0

                while len(active_jobs) < max_jobs_per_node:
                    active_cpu = _active_cpu_alloc(active_jobs, jobs_by_id)
                    active_gpu = _active_gpu_alloc(active_jobs, jobs_by_id)
                    available_cpu = float(candidate.reclaimable_cores) - active_cpu
                    available_gpu = float(candidate.reclaimable_gpus) - active_gpu
                    if available_cpu <= 0 and available_gpu <= 0:
                        break

                    feasible_window = _feasible_window(
                        waiting_queue=waiting_queue,
                        jobs_by_id=jobs_by_id,
                        reclaimable_cores=max(available_cpu, 0.0),
                        reclaimable_gpus=max(available_gpu, 0.0),
                        excluded_job_ids=excluded_job_ids,
                        feasible_size=feasible_size,
                        scan_limit=scan_limit,
                    )
                    selected_job_id = _select_best_fit_job(
                        feasible_window=feasible_window,
                        jobs_by_id=jobs_by_id,
                        reclaimable_cores=max(available_cpu, 0.0),
                        reclaimable_gpus=max(available_gpu, 0.0),
                        excluded_job_ids=excluded_job_ids,
                    )
                    if selected_job_id is None:
                        break

                    _remove_waiting_job_from_scanned_prefix(
                        waiting_queue,
                        waiting_job_ids,
                        selected_job_id,
                        scan_limit=scan_limit,
                    )
                    active_job_ids.add(selected_job_id)
                    excluded_job_ids.add(selected_job_id)
                    active_jobs.append(selected_job_id)
                    unique_jobs_run.add(selected_job_id)
                    injection_count += 1
                    injections_this_node += 1
                    node_events[node_id]["injected_this_interval"] = True

                    job = jobs_by_id[selected_job_id]
                    job.remaining_seconds = max(job.remaining_seconds - INTERVAL_SECONDS, 0)
                    if job.remaining_seconds == 0:
                        completion_count += 1
                        node_events[node_id]["completed_this_interval"] = True
                        active_jobs.remove(selected_job_id)
                        active_job_ids.discard(selected_job_id)
                        completed_job_ids.add(selected_job_id)

                if active_jobs:
                    active_by_node[node_id] = active_jobs
                else:
                    active_by_node.pop(node_id, None)

        if deferred_preempted:
            deferred_requeue = deferred_preempted

        event_node_ids = set(active_by_node.keys())
        if candidate_lookup is not None:
            event_node_ids.update(candidate_lookup.index.astype(str).tolist())
        event_node_ids.update(node_events.keys())

        if event_node_ids:
            records: list[dict[str, object]] = []
            for node_id in sorted(event_node_ids):
                state_row = state_lookup.loc[node_id] if state_lookup is not None and node_id in state_lookup.index else None
                candidate_row = candidate_lookup.loc[node_id] if candidate_lookup is not None and node_id in candidate_lookup.index else None
                active_jobs = list(active_by_node.get(node_id, []))
                active_cpu = _active_cpu_alloc(active_jobs, jobs_by_id)
                active_gpu = _active_gpu_alloc(active_jobs, jobs_by_id)
                dominant_job_id = _dominant_job(active_jobs, jobs_by_id)
                dominant_remaining = jobs_by_id[dominant_job_id].remaining_seconds if dominant_job_id is not None else pd.NA
                cpu_util = float(state_row.cpu_util) if state_row is not None else pd.NA
                gpu_util = float(state_row.gpu_util) if state_row is not None else pd.NA
                memory_util = float(state_row.memory_util) if state_row is not None else pd.NA
                total_util = (
                    float(cpu_util) + (active_cpu / CPU_CORES_PER_NODE)
                    if pd.notna(cpu_util)
                    else pd.NA
                )
                records.append(
                    {
                        "timestamp": current_ts,
                        "node_id": node_id,
                        "mode": mode,
                        "threshold": threshold,
                        "allowed_total_util": primary_reserve_util,
                        "cpu_util": cpu_util,
                        "gpu_util": gpu_util,
                        "memory_util": memory_util,
                        "is_underutilized": candidate_row is not None,
                        "reclaimable_cores": (
                            int(candidate_row.reclaimable_cores)
                            if candidate_row is not None
                            else int(state_row.reclaimable_cores_now) if state_row is not None else 0
                        ),
                        "reclaimable_gpus": (
                            int(candidate_row.reclaimable_gpus)
                            if candidate_row is not None
                            else int(state_row.reclaimable_gpus_now) if state_row is not None else 0
                        ),
                        "opportunistic_job_id": dominant_job_id if dominant_job_id is not None else pd.NA,
                        "opportunistic_job_count": int(len(active_jobs)),
                        "opportunistic_job_ids_json": json.dumps(sorted(active_jobs)) if active_jobs else "[]",
                        "opportunistic_cpu_allocated": active_cpu,
                        "opportunistic_gpu_allocated": active_gpu,
                        "opportunistic_remaining_seconds": dominant_remaining,
                        "injected_this_interval": node_events[node_id]["injected_this_interval"],
                        "preempted_this_interval": node_events[node_id]["preempted_this_interval"],
                        "completed_this_interval": node_events[node_id]["completed_this_interval"],
                        "total_util": total_util,
                    }
                )

            batch = pd.DataFrame(records).sort_values(["timestamp", "node_id"]).reset_index(drop=True)
            batch["opportunistic_job_id"] = batch["opportunistic_job_id"].astype("string")
            batch["opportunistic_remaining_seconds"] = batch["opportunistic_remaining_seconds"].astype("Float64")
            log_batches.append(batch)
        if checkpoint_interval_timestamps > 0 and (ts_idx % checkpoint_interval_timestamps == 0):
            next_batch_idx = _flush_log_batches(log_batches, batch_dir, next_batch_idx)
            checkpoint_payload = _checkpoint_payload(
                next_timestamp_index=ts_idx,
                queue_pointer=queue_pointer,
                waiting_queue=waiting_queue,
                waiting_job_ids=waiting_job_ids,
                deferred_requeue=deferred_requeue,
                active_by_node=active_by_node,
                active_job_ids=active_job_ids,
                completed_job_ids=completed_job_ids,
                unique_jobs_run=unique_jobs_run,
                injection_count=injection_count,
                preemption_count=preemption_count,
                completion_count=completion_count,
                pooling_timestamps_checked=pooling_timestamps_checked,
                pooling_improvement_timestamps=pooling_improvement_timestamps,
                previous_ts=previous_ts,
                next_batch_idx=next_batch_idx,
                jobs_by_id=jobs_by_id,
            )
            _write_checkpoint(checkpoint_path, checkpoint_payload)
        if ts_idx % 250 == 0:
            print(
                (
                    f"phase9 progress scenario={scenario} timestamps={ts_idx}/{len(timeline)} "
                    f"waiting={len(waiting_queue)} active_nodes={len(active_by_node)} "
                    f"injections={injection_count} preemptions={preemption_count} completions={completion_count}"
                ),
                flush=True,
            )

    next_batch_idx = _flush_log_batches(log_batches, batch_dir, next_batch_idx)
    rows_written = _finalize_simulation_output(batch_dir, output_path)
    simulation_log = pd.read_parquet(output_path)
    injection_count_logged = int(simulation_log["injected_this_interval"].sum()) if not simulation_log.empty else 0
    preemption_count_logged = int(simulation_log["preempted_this_interval"].sum()) if not simulation_log.empty else 0
    completion_count_logged = int(simulation_log["completed_this_interval"].sum()) if not simulation_log.empty else 0
    unique_jobs_logged = int(len(unique_jobs_run))

    summary = {
        "status": "complete",
        "simulation_log_path": str(output_path),
        "rows_written": int(rows_written),
        "coverage_start": None if simulation_log.empty else simulation_log["timestamp"].min().isoformat(),
        "coverage_end": None if simulation_log.empty else simulation_log["timestamp"].max().isoformat(),
        "node_count": int(simulation_log["node_id"].nunique()) if not simulation_log.empty else 0,
        "timestamp_count": int(simulation_log["timestamp"].nunique()) if not simulation_log.empty else 0,
        "mode": mode,
        "threshold": threshold,
        "allowed_total_util": primary_reserve_util,
        "primary_reserve_util": primary_reserve_util,
        "injection_count": injection_count_logged,
        "preemption_count": preemption_count_logged,
        "completion_count": completion_count_logged,
        "unique_opportunistic_jobs_run": unique_jobs_logged,
        "pooling_improvement_timestamps": int(pooling_improvement_timestamps),
        "pooling_timestamps_checked": int(pooling_timestamps_checked),
        "columns": simulation_log.columns.tolist(),
    }
    summary_path = output_dir / "simulation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return summary
