from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hpc_sim.config import load_config, resolve_repo_path
from src.hpc_sim.pipeline import scenario_matrix


def _load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _phase10_summary(config: dict) -> pd.DataFrame:
    results_root = resolve_repo_path(config, config["results_dir"])
    rows: list[dict[str, object]] = []
    for scenario in scenario_matrix(config):
        payload = _load_payload(results_root / scenario / "phase_10_metrics" / "payload.json")
        rows.append(payload)
    frame = pd.DataFrame(rows)
    frame["threshold"] = pd.to_numeric(frame["threshold"], errors="coerce")
    return frame


def validate_scenarios(config: dict) -> tuple[pd.DataFrame, dict[str, object]]:
    results_root = resolve_repo_path(config, config["results_dir"])
    cpu_cap = float(config["cluster"]["cpu_cores_per_node"])
    job_cap = int(config["queue"].get("max_injected_jobs_per_node", 1))

    rows: list[dict[str, object]] = []
    for scenario in scenario_matrix(config):
        phase_09_payload = _load_payload(results_root / scenario / "phase_09_simulation_loop" / "payload.json")
        simulation_log_path = Path(phase_09_payload["simulation_log_path"])
        log = pd.read_parquet(
            simulation_log_path,
            columns=[
                "timestamp",
                "node_id",
                "opportunistic_cpu_allocated",
                "opportunistic_gpu_allocated",
                "opportunistic_job_count",
                "preempted_this_interval",
            ],
        )
        log["timestamp"] = pd.to_datetime(log["timestamp"])
        log["opportunistic_cpu_allocated"] = pd.to_numeric(log["opportunistic_cpu_allocated"], errors="coerce").fillna(0.0)
        log["opportunistic_gpu_allocated"] = pd.to_numeric(log["opportunistic_gpu_allocated"], errors="coerce").fillna(0.0)
        log["opportunistic_job_count"] = pd.to_numeric(log["opportunistic_job_count"], errors="coerce").fillna(0).astype("int64")
        ordered = log.sort_values(["timestamp", "node_id"], kind="mergesort").reset_index(drop=True)
        timestamps_monotonic = ordered[["timestamp", "node_id"]].equals(log[["timestamp", "node_id"]].reset_index(drop=True))

        preempted = log["preempted_this_interval"].fillna(False)
        preemption_restore_ok = True
        if bool(preempted.any()):
            preemption_restore_ok = bool(
                (
                    (log.loc[preempted, "opportunistic_cpu_allocated"] >= -1e-9)
                    & (log.loc[preempted, "opportunistic_gpu_allocated"] >= -1e-9)
                    & (log.loc[preempted, "opportunistic_job_count"] >= 0)
                    & (log.loc[preempted, "opportunistic_job_count"] <= job_cap)
                ).all()
            )

        rows.append(
            {
                "scenario": scenario,
                "cpu_within_node_cap": bool((log["opportunistic_cpu_allocated"] <= cpu_cap + 1e-9).all()),
                "job_count_within_configured_cap": bool((log["opportunistic_job_count"] <= job_cap).all()),
                "no_negative_resources": bool(
                    (
                        (log["opportunistic_cpu_allocated"] >= -1e-9)
                        & (log["opportunistic_gpu_allocated"] >= -1e-9)
                        & (log["opportunistic_job_count"] >= 0)
                    ).all()
                ),
                "timestamps_monotonic": timestamps_monotonic,
                "preemption_restores_state_partial": preemption_restore_ok,
                "max_opportunistic_cpu_allocated": float(log["opportunistic_cpu_allocated"].max()),
                "max_opportunistic_gpu_allocated": float(log["opportunistic_gpu_allocated"].max()),
                "max_opportunistic_job_count": int(log["opportunistic_job_count"].max()),
            }
        )

    table = pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)

    phase10 = _phase10_summary(config)
    phase10_nonbaseline = phase10.loc[phase10["mode"] != "original"].copy()
    sanity_rows: list[dict[str, object]] = []
    for threshold, frame in phase10.groupby("threshold", sort=True):
        baseline = frame.loc[frame["mode"] == "original"].iloc[0]
        aggressive = frame.loc[frame["mode"] == "aggressive"].iloc[0]
        buffered = frame.loc[frame["mode"] == "buffered"].iloc[0]
        sanity_rows.append(
            {
                "threshold": float(threshold),
                "buffered_lt_aggressive_interference": bool(buffered["interference_event_count"] < aggressive["interference_event_count"]),
                "aggressive_ge_buffered_ge_original_utilization": bool(
                    (aggressive["mean_total_cpu_util"] >= buffered["mean_total_cpu_util"])
                    and (buffered["mean_total_cpu_util"] >= baseline["mean_total_cpu_util"])
                ),
            }
        )

    preemption_growth = []
    for mode, frame in phase10_nonbaseline.groupby("mode", sort=True):
        ordered = frame.sort_values("threshold")
        values = ordered["preemption_count"].tolist()
        preemption_growth.append(
            {
                "mode": mode,
                "preemptions_increase_with_threshold": bool(all(values[idx] <= values[idx + 1] for idx in range(len(values) - 1))),
            }
        )

    overall = {
        "scenario_checks": rows,
        "sanity_constraints_by_threshold": sanity_rows,
        "preemption_growth_by_mode": preemption_growth,
        "configured_job_cap": job_cap,
        "note": (
            "Validation enforces the final configured opportunistic-job cap per node. "
            "The final simulator intentionally allows up to max_injected_jobs_per_node jobs per node. "
            "GPU totals are reported as logical allocated demand in the final model, so the auto-check focuses "
            "on non-negativity and configured job-count caps rather than a strict per-row GPU-device ceiling."
        ),
    }
    return table, overall


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lightweight validation checks over final pipeline outputs.")
    parser.add_argument("--config", default="config/default.json")
    args = parser.parse_args()

    config = load_config(args.config)
    outputs_dir = resolve_repo_path(config, "outputs")
    tables_dir = outputs_dir / "tables"
    logs_dir = outputs_dir / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    table, overall = validate_scenarios(config)
    csv_path = tables_dir / "validation_checks.csv"
    json_path = logs_dir / "validation_summary.json"
    table.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"validation_csv": str(csv_path), "validation_json": str(json_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
