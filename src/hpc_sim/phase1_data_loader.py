from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import resolve_repo_path

CPU_REQUIRED_COLUMNS = [
    "node_id",
    "timestamp",
    "cpu1_temp",
    "cpu2_temp",
    "cpu1_frequency",
    "cpu2_frequency",
    "cpu1_utilization",
    "cpu2_utilization",
]

GPU_REQUIRED_COLUMNS = [
    "card_id",
    "timestamp",
    "is_healthy",
    "power_consum",
    "utilization",
    "memory",
    "temperature",
]

MEMORY_REQUIRED_COLUMNS = [
    "node_id",
    "timestamp",
    "memory_used_percent",
    "memory_used_mb",
    "swap_used_percent",
    "page_faults",
    "memory_bandwidth",
]

JOB_REQUIRED_COLUMNS = [
    "job_id",
    "ctime",
    "qtime",
    "stime",
    "etime",
    "mtime",
    "exec_host",
    "exec_vnode",
]

JOB_TIMESTAMP_COLUMNS = ["ctime", "qtime", "stime", "etime", "mtime"]
NODE_PATTERN = re.compile(r"r\d+(?:c|g)n\d+")
GPU_CARD_PATTERN = re.compile(r"^(?P<rack>\d+)_(?P<slot>\d+)_(?P<card>\d+)$")


@dataclass
class CoverageTracker:
    minimum: pd.Timestamp | None = None
    maximum: pd.Timestamp | None = None

    def update(self, values: pd.Series) -> None:
        non_null = values.dropna()
        if non_null.empty:
            return
        current_min = non_null.min()
        current_max = non_null.max()
        if self.minimum is None or current_min < self.minimum:
            self.minimum = current_min
        if self.maximum is None or current_max > self.maximum:
            self.maximum = current_max

    def to_dict(self) -> dict[str, str | None]:
        return {
            "start": None if self.minimum is None else self.minimum.isoformat(),
            "end": None if self.maximum is None else self.maximum.isoformat(),
        }


def _serialize_scalar(value) -> str | pd._libs.missing.NAType:
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    if pd.isna(value):
        return pd.NA
    return str(value)


def _require_columns(columns: Iterable[str], required: list[str], dataset_name: str) -> None:
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"{dataset_name} missing required columns: {missing}")


def _normalize_node_id(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("", pd.NA)
    )


def _normalize_gpu_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    card_parts = chunk["card_id"].astype(str).str.extract(GPU_CARD_PATTERN)
    if card_parts.isna().any(axis=None):
        bad_examples = chunk.loc[card_parts.isna().any(axis=1), "card_id"].head(5).tolist()
        raise ValueError(f"Unexpected GPU card_id format: {bad_examples}")

    chunk = chunk.copy()
    chunk["card_id_raw"] = chunk["card_id"]
    chunk["card_id"] = chunk["card_id"].astype(str).str.strip().str.lower()
    chunk["node_id"] = (
        "r"
        + card_parts["rack"].astype(str).str.zfill(2)
        + "gn"
        + card_parts["slot"].astype(str).str.zfill(2)
    )
    chunk["gpu_index"] = pd.to_numeric(card_parts["card"], errors="coerce").astype("Int64")
    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True)
    chunk["is_healthy"] = chunk["is_healthy"].astype(str).str.lower().map({"true": True, "false": False})
    for column in ["power_consum", "utilization", "memory", "temperature"]:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
    return chunk


def _normalize_memory_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["node_id_raw"] = chunk["node_id"]
    chunk["node_id"] = _normalize_node_id(chunk["node_id"])
    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True)
    for column in MEMORY_REQUIRED_COLUMNS[2:]:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
    return chunk


def _normalize_cpu_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["node_id_raw"] = chunk["node_id"]
    chunk["node_id"] = _normalize_node_id(chunk["node_id"])
    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True)
    for column in CPU_REQUIRED_COLUMNS[2:]:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
    chunk["cpu_utilization_mean"] = chunk[["cpu1_utilization", "cpu2_utilization"]].mean(axis=1)
    return chunk


def _extract_job_nodes(exec_vnode: str, exec_host: str) -> list[str]:
    values: list[str] = []
    for raw_value in [exec_vnode, exec_host]:
        if pd.isna(raw_value):
            continue
        value = str(raw_value).strip()
        if value:
            values.append(value)
    candidates = " ".join(values)
    seen: list[str] = []
    for node_id in NODE_PATTERN.findall(candidates.lower()):
        if node_id not in seen:
            seen.append(node_id)
    return seen


def _normalize_job_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.copy()
    for column in JOB_TIMESTAMP_COLUMNS:
        chunk[f"{column}_ts"] = pd.to_datetime(
            chunk[column],
            format="%a %b %d %H:%M:%S %Y",
            errors="coerce",
            utc=True,
        )
    chunk["job_earliest_ts"] = chunk[[f"{column}_ts" for column in JOB_TIMESTAMP_COLUMNS]].min(axis=1)
    chunk["job_latest_ts"] = chunk[[f"{column}_ts" for column in JOB_TIMESTAMP_COLUMNS]].max(axis=1)

    node_lists = [
        _extract_job_nodes(exec_vnode, exec_host)
        for exec_vnode, exec_host in zip(chunk["exec_vnode"], chunk["exec_host"])
    ]
    chunk["node_ids_json"] = [json.dumps(nodes) for nodes in node_lists]
    chunk["primary_node_id"] = [nodes[0] if nodes else pd.NA for nodes in node_lists]
    chunk["node_count"] = [len(nodes) for nodes in node_lists]

    chunk["node_count"] = pd.to_numeric(chunk["node_count"], errors="coerce").astype("Int64")

    preserved_columns = set([f"{column}_ts" for column in JOB_TIMESTAMP_COLUMNS] + ["job_earliest_ts", "job_latest_ts", "node_count"])
    for column in chunk.columns:
        if column in preserved_columns:
            continue
        chunk[column] = chunk[column].apply(_serialize_scalar).astype("string")
    return chunk


def _write_chunks(
    chunk_iter: Iterable[pd.DataFrame],
    output_path: Path,
    normalizer,
    dataset_name: str,
    timestamp_column: str,
) -> dict[str, object]:
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    chunk_count = 0
    coverage = CoverageTracker()
    schema: pa.Schema | None = None

    try:
        for chunk in chunk_iter:
            chunk_count += 1
            normalized = normalizer(chunk)
            if normalized.empty:
                continue

            if normalized[timestamp_column].isna().all():
                raise ValueError(f"{dataset_name} has no parseable timestamps in chunk {chunk_count}")

            coverage.update(normalized[timestamp_column])
            table = pa.Table.from_pandas(normalized, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(output_path, schema=schema, compression="snappy")
            else:
                table = table.cast(schema)

            writer.write_table(table)
            rows_written += len(normalized)
    finally:
        if writer is not None:
            writer.close()

    return {
        "rows_written": rows_written,
        "chunks_processed": chunk_count,
        "coverage": coverage.to_dict(),
        "output_path": str(output_path),
    }


def _jsonl_chunk_reader(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    buffer: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            buffer.append(json.loads(line))
            if len(buffer) >= chunksize:
                yield pd.DataFrame(buffer)
                buffer = []
    if buffer:
        yield pd.DataFrame(buffer)


def _collect_jsonl_columns(path: Path) -> list[str]:
    columns: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            columns.update(json.loads(line).keys())
    return sorted(columns)


def _jsonl_chunk_reader_with_schema(path: Path, chunksize: int, columns: list[str]) -> Iterable[pd.DataFrame]:
    for chunk in _jsonl_chunk_reader(path, chunksize):
        yield chunk.reindex(columns=columns)


def run_phase_01(config: dict) -> dict[str, object]:
    processed_dir = resolve_repo_path(config, config["processed_data_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    cpu_path = resolve_repo_path(config, config["files"]["cpu_metrics"])
    gpu_path = resolve_repo_path(config, config["files"]["gpu_metrics"])
    memory_path = resolve_repo_path(config, config["files"]["memory_metrics"])
    jobs_path = resolve_repo_path(config, config["files"]["jobs"])

    cpu_preview = pd.read_csv(cpu_path, nrows=5)
    gpu_preview = pd.read_csv(gpu_path, nrows=5)
    memory_preview = pd.read_csv(memory_path, nrows=5)
    job_columns = _collect_jsonl_columns(jobs_path)
    jobs_preview = next(_jsonl_chunk_reader_with_schema(jobs_path, 5, job_columns))

    _require_columns(cpu_preview.columns, CPU_REQUIRED_COLUMNS, "cpu_metrics.csv")
    _require_columns(gpu_preview.columns, GPU_REQUIRED_COLUMNS, "gpu_status.csv")
    _require_columns(memory_preview.columns, MEMORY_REQUIRED_COLUMNS, "memory_data.csv")
    _require_columns(jobs_preview.columns, JOB_REQUIRED_COLUMNS, "merged_all_jobs.jsonl")

    cpu_output = processed_dir / "cpu_raw.parquet"
    gpu_output = processed_dir / "gpu_raw.parquet"
    memory_output = processed_dir / "memory_raw.parquet"
    job_output = processed_dir / "job_raw.parquet"
    summary_output = processed_dir / "phase_01_summary.json"

    cpu_result = _write_chunks(
        pd.read_csv(cpu_path, chunksize=config["phase_01"]["cpu_chunksize"]),
        cpu_output,
        _normalize_cpu_chunk,
        "cpu_metrics.csv",
        "timestamp",
    )
    gpu_result = _write_chunks(
        pd.read_csv(gpu_path, chunksize=config["phase_01"]["gpu_chunksize"]),
        gpu_output,
        _normalize_gpu_chunk,
        "gpu_status.csv",
        "timestamp",
    )
    memory_result = _write_chunks(
        pd.read_csv(memory_path, chunksize=config["phase_01"]["gpu_chunksize"]),
        memory_output,
        _normalize_memory_chunk,
        "memory_data.csv",
        "timestamp",
    )
    jobs_result = _write_chunks(
        _jsonl_chunk_reader_with_schema(jobs_path, config["phase_01"]["job_chunksize"], job_columns),
        job_output,
        _normalize_job_chunk,
        "merged_all_jobs.jsonl",
        "job_earliest_ts",
    )

    summary = {
        "seed": config["seed"],
        "assumptions": {
            "job_input_source": "Data/merged_all_jobs.jsonl",
            "gpu_input_source": "Data/gpu_status.csv",
            "memory_input_source": "Data/memory_data.csv",
            "processed_output_dir": str(processed_dir),
            "node_normalization": "lowercase canonical node ids; GPU node_id derived from card_id; job nodes extracted from exec_vnode/exec_host",
            "timestamp_parsing": {
                "cpu_gpu": "parsed from source strings and normalized to UTC-aware timestamps",
                "jobs": "parsed from PBS textual timestamps and normalized to UTC-aware timestamps",
                "memory": "parsed from source strings and normalized to UTC-aware timestamps",
            },
        },
        "validated_schemas": {
            "cpu_metrics.csv": CPU_REQUIRED_COLUMNS,
            "gpu_status.csv": GPU_REQUIRED_COLUMNS,
            "memory_data.csv": MEMORY_REQUIRED_COLUMNS,
            "merged_all_jobs.jsonl": JOB_REQUIRED_COLUMNS,
        },
        "job_column_count": len(job_columns),
        "outputs": {
            "cpu_raw_parquet": cpu_result,
            "gpu_raw_parquet": gpu_result,
            "memory_raw_parquet": memory_result,
            "job_raw_parquet": jobs_result,
        },
        "source_date_ranges": {
            "cpu_metrics.csv": cpu_result["coverage"],
            "gpu_status.csv": gpu_result["coverage"],
            "memory_data.csv": memory_result["coverage"],
            "merged_all_jobs.jsonl": jobs_result["coverage"],
        },
    }

    with summary_output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return summary
