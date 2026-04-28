from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import load_config, resolve_repo_path
from .phase1_data_loader import run_phase_01
from .phase2_preprocessing import run_phase_02
from .phase3_state_builder import run_phase_03
from .phase4_detection import run_phase_04
from .phase5_reclamation import run_phase_05
from .phase6_opportunistic_job_model import run_phase_06
from .phase7_injection import run_phase_07
from .phase8_preemption import run_phase_08
from .phase9_simulation_loop import run_phase_09
from .phase10_metrics import run_phase_10
from .phase11_plotting import run_phase_11


PHASES: list[tuple[str, str]] = [
    ("phase_01_data_loading", "Data Loading"),
    ("phase_02_preprocessing", "Preprocessing"),
    ("phase_03_state_construction", "State Construction"),
    ("phase_04_detection", "Detection"),
    ("phase_05_reclamation", "Reclamation"),
    ("phase_06_opportunistic_job_modeling", "Opportunistic Job Modeling"),
    ("phase_07_injection", "Injection"),
    ("phase_08_preemption", "Preemption"),
    ("phase_09_simulation_loop", "Simulation Loop"),
    ("phase_10_metrics", "Metrics"),
    ("phase_11_plotting", "Plotting"),
]


@dataclass(frozen=True)
class PhaseDefinition:
    phase_id: str
    display_name: str
    handler: Callable[[dict, Path, Path, str], dict]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_file_fingerprint(path: Path, sample_bytes: int = 1024 * 1024) -> dict:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_epoch": stat.st_mtime,
        "sha256_prefix_first_mb": digest.hexdigest(),
    }


def ensure_phase_input(previous_phase_dir: Path | None) -> None:
    if previous_phase_dir and not previous_phase_dir.exists():
        raise FileNotFoundError(
            f"Missing prerequisite phase output: {previous_phase_dir}"
        )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def stage_stub(config: dict, input_dir: Path, output_dir: Path, scenario: str) -> dict:
    return {
        "status": "placeholder",
        "scenario": scenario,
        "message": (
            "Step 0 scaffold created. Phase implementation will be added in later steps."
        ),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": config["seed"],
    }


def phase_01_data_loading(config: dict, input_dir: Path, output_dir: Path, scenario: str) -> dict:
    file_map = config["files"]
    fingerprints = {
        name: compute_file_fingerprint(resolve_repo_path(config, relative_path))
        for name, relative_path in file_map.items()
    }
    phase_summary = run_phase_01(config)
    return {
        "status": "complete",
        "scenario": scenario,
        "seed": config["seed"],
        "evaluation_interval_minutes": config["telemetry"]["evaluation_interval_minutes"],
        "input_files": fingerprints,
        "phase_01_summary_path": str(
            resolve_repo_path(config, config["processed_data_dir"]) / "phase_01_summary.json"
        ),
        "processed_outputs": phase_summary["outputs"],
        "source_date_ranges": phase_summary["source_date_ranges"],
    }


def phase_02_preprocessing(config: dict, input_dir: Path, output_dir: Path, scenario: str) -> dict:
    phase_summary = run_phase_02(config)
    return {
        "status": "complete",
        "scenario": scenario,
        "seed": config["seed"],
        "phase_02_summary_path": str(
            resolve_repo_path(config, config["processed_data_dir"]) / "phase_02_summary.json"
        ),
        "processed_outputs": phase_summary["outputs"],
        "job_mapping_stats": phase_summary["job_mapping_stats"],
    }


def phase_03_state_construction(config: dict, input_dir: Path, output_dir: Path, scenario: str) -> dict:
    phase_summary = run_phase_03(input_dir, output_dir)
    return {
        "status": "complete",
        "scenario": scenario,
        "seed": config["seed"],
        "state_cache_path": phase_summary["state_cache_path"],
        "rows_written": phase_summary["rows_written"],
        "columns": phase_summary["columns"],
        "coverage_start": phase_summary["coverage_start"],
        "coverage_end": phase_summary["coverage_end"],
        "node_count": phase_summary["node_count"],
        "timestamp_count": phase_summary["timestamp_count"],
        "explicit_missing_telemetry_rows": phase_summary["explicit_missing_telemetry_rows"],
        "metrics_missing_after_ffill": phase_summary["metrics_missing_after_ffill"],
        "active_job_stats": phase_summary["active_job_stats"],
        "multi_job_resolution_rule": phase_summary["multi_job_resolution_rule"],
    }


def phase_04_detection(config: dict, input_dir: Path, output_dir: Path, scenario: str) -> dict:
    phase_summary = run_phase_04(input_dir, output_dir, scenario)
    return {
        "status": "complete",
        "scenario": scenario,
        "seed": config["seed"],
        "threshold": phase_summary["threshold"],
        "underutilized_nodes_path": phase_summary["underutilized_nodes_path"],
        "rows_written": phase_summary["rows_written"],
        "underutilized_rows": phase_summary["underutilized_rows"],
        "cpu_underutilized_rows": phase_summary["cpu_underutilized_rows"],
        "gpu_underutilized_rows": phase_summary["gpu_underutilized_rows"],
        "eligible_rows_with_previous_interval": phase_summary["eligible_rows_with_previous_interval"],
        "memory_spike_rows": phase_summary["memory_spike_rows"],
        "coverage_start": phase_summary["coverage_start"],
        "coverage_end": phase_summary["coverage_end"],
        "node_count": phase_summary["node_count"],
        "timestamp_count": phase_summary["timestamp_count"],
        "columns": phase_summary["columns"],
    }


def phase_05_reclamation(config: dict, input_dir: Path, output_dir: Path, scenario: str) -> dict:
    scenario_dir = output_dir.parent
    phase_summary = run_phase_05(input_dir, scenario_dir, output_dir)
    return {
        "status": "complete",
        "scenario": scenario,
        "seed": config["seed"],
        "reclamation_plan_path": phase_summary["reclamation_plan_path"],
        "rows_written": phase_summary["rows_written"],
        "coverage_start": phase_summary["coverage_start"],
        "coverage_end": phase_summary["coverage_end"],
        "node_count": phase_summary["node_count"],
        "timestamp_count": phase_summary["timestamp_count"],
        "rows_with_cpu_reclamation": phase_summary["rows_with_cpu_reclamation"],
        "rows_with_gpu_reclamation": phase_summary["rows_with_gpu_reclamation"],
        "max_reclaimable_cores": phase_summary["max_reclaimable_cores"],
        "max_reclaimable_gpus": phase_summary["max_reclaimable_gpus"],
        "columns": phase_summary["columns"],
        "gpu_reclamation_rule": phase_summary["gpu_reclamation_rule"],
    }


def phase_06_opportunistic_job_modeling(
    config: dict, input_dir: Path, output_dir: Path, scenario: str
) -> dict:
    phase_summary = run_phase_06(config, input_dir, output_dir)
    return {
        "status": "complete",
        "scenario": scenario,
        "seed": config["seed"],
        "opportunistic_job_pool_path": phase_summary["opportunistic_job_pool_path"],
        "rows_written": phase_summary["rows_written"],
        "coverage_start": phase_summary["coverage_start"],
        "coverage_end": phase_summary["coverage_end"],
        "node_count": phase_summary["node_count"],
        "timestamp_count": phase_summary["timestamp_count"],
        "rows_with_candidate_fit": phase_summary["rows_with_candidate_fit"],
        "rows_without_candidate_fit": phase_summary["rows_without_candidate_fit"],
        "unique_selected_jobs": phase_summary["unique_selected_jobs"],
        "max_selection_window_size": phase_summary["max_selection_window_size"],
        "max_fit_candidate_count": phase_summary["max_fit_candidate_count"],
        "columns": phase_summary["columns"],
        "selection_policy": phase_summary["selection_policy"],
        "duration_remaining_rule": phase_summary["duration_remaining_rule"],
    }


def phase_07_injection(config: dict, input_dir: Path, output_dir: Path, scenario: str) -> dict:
    scenario_dir = output_dir.parent
    phase_summary = run_phase_07(input_dir, scenario_dir, output_dir)
    return {
        "status": "complete",
        "scenario": scenario,
        "seed": config["seed"],
        "injection_events_path": phase_summary["injection_events_path"],
        "state_after_injection_path": phase_summary["state_after_injection_path"],
        "rows_written": phase_summary["rows_written"],
        "coverage_start": phase_summary["coverage_start"],
        "coverage_end": phase_summary["coverage_end"],
        "node_count": phase_summary["node_count"],
        "timestamp_count": phase_summary["timestamp_count"],
        "rows_with_candidate_fit": phase_summary["rows_with_candidate_fit"],
        "successful_injections": phase_summary["successful_injections"],
        "rows_without_injection": phase_summary["rows_without_injection"],
        "unique_injected_jobs": phase_summary["unique_injected_jobs"],
        "max_post_injection_allocated_cores": phase_summary["max_post_injection_allocated_cores"],
        "max_post_injection_allocated_gpus": phase_summary["max_post_injection_allocated_gpus"],
        "columns": phase_summary["columns"],
        "state_columns": phase_summary["state_columns"],
        "injection_rule": phase_summary["injection_rule"],
    }


def phase_08_preemption(config: dict, input_dir: Path, output_dir: Path, scenario: str) -> dict:
    phase_summary = run_phase_08(config, input_dir, output_dir)
    return {
        "status": "complete",
        "scenario": scenario,
        "seed": config["seed"],
        "preemption_events_path": phase_summary["preemption_events_path"],
        "state_after_preemption_path": phase_summary["state_after_preemption_path"],
        "rows_written": phase_summary["rows_written"],
        "coverage_start": phase_summary["coverage_start"],
        "coverage_end": phase_summary["coverage_end"],
        "node_count": phase_summary["node_count"],
        "timestamp_count": phase_summary["timestamp_count"],
        "rows_with_opportunistic_job_before_preemption": phase_summary["rows_with_opportunistic_job_before_preemption"],
        "preemption_count": phase_summary["preemption_count"],
        "rows_with_opportunistic_job_after_preemption": phase_summary["rows_with_opportunistic_job_after_preemption"],
        "max_total_util": phase_summary["max_total_util"],
        "max_allowed_total_util": phase_summary["max_allowed_total_util"],
        "columns": phase_summary["columns"],
        "state_columns": phase_summary["state_columns"],
        "preemption_rule": phase_summary["preemption_rule"],
    }


def phase_09_simulation_loop(config: dict, input_dir: Path, output_dir: Path, scenario: str) -> dict:
    scenario_dir = output_dir.parent
    phase_summary = run_phase_09(config, input_dir, scenario_dir, output_dir, scenario)
    return {
        "status": "complete",
        "scenario": scenario,
        "seed": config["seed"],
        "simulation_log_path": phase_summary["simulation_log_path"],
        "rows_written": phase_summary["rows_written"],
        "coverage_start": phase_summary["coverage_start"],
        "coverage_end": phase_summary["coverage_end"],
        "node_count": phase_summary["node_count"],
        "timestamp_count": phase_summary["timestamp_count"],
        "mode": phase_summary["mode"],
        "threshold": phase_summary["threshold"],
        "allowed_total_util": phase_summary["allowed_total_util"],
        "primary_reserve_util": phase_summary.get("primary_reserve_util", phase_summary["allowed_total_util"]),
        "injection_count": phase_summary["injection_count"],
        "preemption_count": phase_summary["preemption_count"],
        "completion_count": phase_summary["completion_count"],
        "unique_opportunistic_jobs_run": phase_summary["unique_opportunistic_jobs_run"],
        "pooling_improvement_timestamps": phase_summary["pooling_improvement_timestamps"],
        "pooling_timestamps_checked": phase_summary["pooling_timestamps_checked"],
        "columns": phase_summary["columns"],
    }


def phase_10_metrics(config: dict, input_dir: Path, output_dir: Path, scenario: str) -> dict:
    scenario_dir = output_dir.parent
    phase_summary = run_phase_10(config, input_dir, scenario_dir, output_dir, scenario)
    return {
        "status": "complete",
        "scenario": scenario,
        "seed": config["seed"],
        "mode": phase_summary["mode"],
        "threshold": phase_summary["threshold"],
        "allowed_total_util": phase_summary["allowed_total_util"],
        "metrics_summary_path": phase_summary["metrics_summary_path"],
        "metrics_weekly_path": phase_summary["metrics_weekly_path"],
        "rows_evaluated": phase_summary["rows_evaluated"],
        "active_opportunistic_rows": phase_summary["active_opportunistic_rows"],
        "utilization_improvement_points": phase_summary["utilization_improvement_points"],
        "cluster_utilization_improvement_pct": phase_summary["cluster_utilization_improvement_pct"],
        "mean_primary_cpu_util": phase_summary["mean_primary_cpu_util"],
        "mean_total_cpu_util": phase_summary["mean_total_cpu_util"],
        "preemption_count": phase_summary["preemption_count"],
        "interference_event_count": phase_summary["interference_event_count"],
        "completion_count": phase_summary["completion_count"],
        "throughput_proxy_cpu_core_minutes": phase_summary["throughput_proxy_cpu_core_minutes"],
        "throughput_proxy_gpu_device_minutes": phase_summary["throughput_proxy_gpu_device_minutes"],
        "relative_cpu_throughput_gain_fraction": phase_summary["relative_cpu_throughput_gain_fraction"],
        "relative_cpu_throughput_gain_pct": phase_summary["relative_cpu_throughput_gain_pct"],
        "unique_opportunistic_jobs_run": phase_summary["unique_opportunistic_jobs_run"],
        "weekly_windows": phase_summary["weekly_windows"],
        "metric_definitions": phase_summary["metric_definitions"],
        "baseline_reference": phase_summary["baseline_reference"],
    }


def phase_11_plotting(config: dict, input_dir: Path, output_dir: Path, scenario: str) -> dict:
    del input_dir
    phase_summary = run_phase_11(config, output_dir, scenario)
    return {
        "status": "complete",
        "scenario": scenario,
        "seed": config["seed"],
        "plot_output_dir": phase_summary["plot_output_dir"],
        "paper_outputs_dir": phase_summary["paper_outputs_dir"],
        "figures": phase_summary["figures"],
        "paper_output_figures": phase_summary["paper_output_figures"],
        "best_scenario": phase_summary["best_scenario"],
        "best_completion_count": phase_summary["best_completion_count"],
        "best_cpu_core_minutes": phase_summary["best_cpu_core_minutes"],
        "wait_time_plot_available": phase_summary["wait_time_plot_available"],
        "wait_time_plot_reason": phase_summary["wait_time_plot_reason"],
        "recommended_optional_figures": phase_summary["recommended_optional_figures"],
    }


def build_phase_definitions() -> dict[str, PhaseDefinition]:
    definitions: dict[str, PhaseDefinition] = {}
    for phase_id, display_name in PHASES:
        if phase_id == "phase_01_data_loading":
            handler = phase_01_data_loading
        elif phase_id == "phase_02_preprocessing":
            handler = phase_02_preprocessing
        elif phase_id == "phase_03_state_construction":
            handler = phase_03_state_construction
        elif phase_id == "phase_04_detection":
            handler = phase_04_detection
        elif phase_id == "phase_05_reclamation":
            handler = phase_05_reclamation
        elif phase_id == "phase_06_opportunistic_job_modeling":
            handler = phase_06_opportunistic_job_modeling
        elif phase_id == "phase_07_injection":
            handler = phase_07_injection
        elif phase_id == "phase_08_preemption":
            handler = phase_08_preemption
        elif phase_id == "phase_09_simulation_loop":
            handler = phase_09_simulation_loop
        elif phase_id == "phase_10_metrics":
            handler = phase_10_metrics
        elif phase_id == "phase_11_plotting":
            handler = phase_11_plotting
        else:
            handler = stage_stub
        definitions[phase_id] = PhaseDefinition(phase_id, display_name, handler)
    return definitions


def run_phase(config: dict, phase_id: str, scenario: str, force: bool) -> Path:
    phase_definitions = build_phase_definitions()
    if phase_id not in phase_definitions:
        raise KeyError(f"Unknown phase: {phase_id}")

    phase_index = [idx for idx, (pid, _) in enumerate(PHASES) if pid == phase_id][0]
    previous_phase_id = PHASES[phase_index - 1][0] if phase_index > 0 else None

    results_root = resolve_repo_path(config, config["results_dir"])
    scenario_dir = results_root / scenario
    phase_dir = scenario_dir / phase_id
    manifest_path = phase_dir / "manifest.json"
    payload_path = phase_dir / "payload.json"
    previous_dir = scenario_dir / previous_phase_id if previous_phase_id else None
    phase_09_checkpoint_path = phase_dir / "_resume" / "checkpoint_state.json"

    if manifest_path.exists() and not force and not (phase_id == "phase_09_simulation_loop" and phase_09_checkpoint_path.exists()):
        return manifest_path

    ensure_phase_input(previous_dir)
    phase_dir.mkdir(parents=True, exist_ok=True)

    definition = phase_definitions[phase_id]
    started_at = utc_now()
    previous_force = os.environ.get("HPC_SIM_FORCE")
    os.environ["HPC_SIM_FORCE"] = "1" if force else "0"
    try:
        payload = definition.handler(config, previous_dir or scenario_dir, phase_dir, scenario)
    finally:
        if previous_force is None:
            os.environ.pop("HPC_SIM_FORCE", None)
        else:
            os.environ["HPC_SIM_FORCE"] = previous_force
    payload["phase_id"] = phase_id
    payload["display_name"] = definition.display_name
    payload["started_at"] = started_at
    payload["completed_at"] = utc_now()

    write_json(payload_path, payload)
    write_json(
        manifest_path,
        {
            "phase_id": phase_id,
            "display_name": definition.display_name,
            "scenario": scenario,
            "status": payload["status"],
            "payload_path": str(payload_path),
            "seed": config["seed"],
            "force": force,
            "completed_at": payload["completed_at"],
        },
    )
    return manifest_path


def scenario_matrix(config: dict) -> list[str]:
    scenarios = []
    for baseline in config["baselines"]:
        for threshold in config["thresholds"]:
            scenarios.append(f"{baseline}_threshold_{threshold:.2f}")
    return scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic HPC simulation pipeline runner.")
    parser.add_argument(
        "--config",
        default="config/default.json",
        help="Path to the pipeline configuration JSON file.",
    )
    parser.add_argument(
        "--phase",
        default="all",
        help="Single phase id to run, or 'all' for the full pipeline.",
    )
    parser.add_argument(
        "--scenario",
        default="all",
        help="Single scenario name, or 'all' for every baseline-threshold scenario.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run phases even if a manifest already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    random.seed(config["seed"])
    os.environ["PYTHONHASHSEED"] = str(config["seed"])

    phase_ids = [phase_id for phase_id, _ in PHASES] if args.phase == "all" else [args.phase]
    scenarios = scenario_matrix(config) if args.scenario == "all" else [args.scenario]

    summary = {
        "seed": config["seed"],
        "config_path": config["_config_path"],
        "phase_ids": phase_ids,
        "scenarios": scenarios,
        "started_at": utc_now(),
        "completed": [],
    }

    for scenario in scenarios:
        for phase_id in phase_ids:
            manifest_path = run_phase(config, phase_id, scenario, args.force)
            summary["completed"].append(
                {
                    "scenario": scenario,
                    "phase_id": phase_id,
                    "manifest_path": str(manifest_path),
                }
            )

    summary["completed_at"] = utc_now()
    summary_path = resolve_repo_path(config, config["results_dir"]) / "last_run_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
