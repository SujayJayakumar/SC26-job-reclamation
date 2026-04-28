from __future__ import annotations

import argparse
import json
import shutil
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


def build_summary_rows(config: dict) -> pd.DataFrame:
    results_root = resolve_repo_path(config, config["results_dir"])
    rows: list[dict[str, object]] = []
    for scenario in scenario_matrix(config):
        payload = _load_payload(results_root / scenario / "phase_10_metrics" / "payload.json")
        rows.append(payload)
    frame = pd.DataFrame(rows)
    frame["threshold"] = pd.to_numeric(frame["threshold"], errors="coerce")
    return frame.sort_values(["threshold", "mode"]).reset_index(drop=True)


def export_tables(summary: pd.DataFrame, tables_dir: Path) -> dict[str, str]:
    tables_dir.mkdir(parents=True, exist_ok=True)

    utilization = summary[
        [
            "scenario",
            "mode",
            "threshold",
            "mean_primary_cpu_util",
            "mean_total_cpu_util",
            "utilization_improvement_points",
            "cluster_utilization_improvement_pct",
            "relative_cpu_throughput_gain_fraction",
            "relative_cpu_throughput_gain_pct",
            "throughput_proxy_cpu_core_minutes",
            "throughput_proxy_gpu_device_minutes",
        ]
    ].copy()
    utilization.to_csv(tables_dir / "utilization.csv", index=False)

    preemptions = summary[
        [
            "scenario",
            "mode",
            "threshold",
            "preemption_count",
            "completion_count",
            "unique_opportunistic_jobs_run",
            "active_opportunistic_rows",
        ]
    ].copy()
    preemptions.to_csv(tables_dir / "preemptions.csv", index=False)

    interference = summary[
        [
            "scenario",
            "mode",
            "threshold",
            "interference_event_count",
            "completion_count",
            "unique_opportunistic_jobs_run",
            "cluster_utilization_improvement_pct",
        ]
    ].copy()
    interference.to_csv(tables_dir / "interference.csv", index=False)

    extras = summary[
        [
            "scenario",
            "mode",
            "threshold",
            "completion_count",
            "unique_opportunistic_jobs_run",
            "throughput_proxy_cpu_core_minutes",
            "throughput_proxy_gpu_device_minutes",
        ]
    ].copy()
    extras.to_csv(tables_dir / "completions.csv", index=False)

    return {
        "utilization_csv": str(tables_dir / "utilization.csv"),
        "preemptions_csv": str(tables_dir / "preemptions.csv"),
        "interference_csv": str(tables_dir / "interference.csv"),
        "completions_csv": str(tables_dir / "completions.csv"),
    }


def export_figures(config: dict, figures_dir: Path) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    source_dir = resolve_repo_path(config, config["paper_outputs_dir"])
    exported: dict[str, str] = {}
    for image_path in sorted(source_dir.glob("*.png")):
        destination = figures_dir / image_path.name
        shutil.copy2(image_path, destination)
        exported[image_path.name] = str(destination)
    return exported


def main() -> int:
    parser = argparse.ArgumentParser(description="Export strict outputs/ tables and figures for AD/AE packaging.")
    parser.add_argument("--config", default="config/default.json")
    args = parser.parse_args()

    config = load_config(args.config)
    outputs_dir = resolve_repo_path(config, "outputs")
    tables_dir = outputs_dir / "tables"
    figures_dir = outputs_dir / "figures"
    logs_dir = outputs_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary_rows(config)
    exported_tables = export_tables(summary, tables_dir)
    exported_figures = export_figures(config, figures_dir)

    manifest = {
        "status": "complete",
        "outputs_dir": str(outputs_dir),
        "tables": exported_tables,
        "figure_count": len(exported_figures),
        "figures_dir": str(figures_dir),
        "logs_dir": str(logs_dir),
    }
    with (outputs_dir / "export_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
