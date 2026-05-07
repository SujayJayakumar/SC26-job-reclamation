from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hpc_sim.config import load_config
from src.hpc_sim.pipeline import PHASES, run_phase, scenario_matrix


PHASE_GROUPS: dict[str, list[str]] = {
    "preprocess": [
        "phase_01_data_loading",
        "phase_02_preprocessing",
        "phase_03_state_construction",
        "phase_04_detection",
        "phase_05_reclamation",
        "phase_06_opportunistic_job_modeling",
        "phase_07_injection",
        "phase_08_preemption",
    ],
    "simulate": ["phase_09_simulation_loop"],
    "metrics": ["phase_10_metrics"],
    "plots": ["phase_11_plotting"],
    "all": [phase_id for phase_id, _ in PHASES],
}

PHASE_NUMBER_TO_ID = {index + 1: phase_id for index, (phase_id, _) in enumerate(PHASES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-level CLI for the HPC opportunistic scheduling simulation.")
    parser.add_argument(
        "--config",
        default="config/default.json",
        help="Path to the pipeline configuration file.",
    )
    parser.add_argument(
        "--phase",
        default="all",
        choices=sorted(PHASE_GROUPS.keys()),
        help="Logical pipeline stage to run.",
    )
    parser.add_argument(
        "--mode",
        choices=["original", "aggressive", "buffered"],
        help="Filter to a single mode.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        choices=[0.15, 0.20, 0.25],
        help="Filter to a single threshold.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run phases even if manifests already exist.",
    )
    parser.add_argument(
        "--start_phase",
        type=int,
        choices=sorted(PHASE_NUMBER_TO_ID.keys()),
        help="Optional inclusive numeric phase range start (for resumable reruns).",
    )
    parser.add_argument(
        "--end_phase",
        type=int,
        choices=sorted(PHASE_NUMBER_TO_ID.keys()),
        help="Optional inclusive numeric phase range end (for resumable reruns).",
    )
    return parser.parse_args()


def select_scenarios(config: dict, mode: str | None, threshold: float | None) -> list[str]:
    scenarios = scenario_matrix(config)
    if mode is None and threshold is None:
        return scenarios

    selected: list[str] = []
    for scenario in scenarios:
        scenario_mode, _, scenario_threshold = scenario.partition("_threshold_")
        if mode is not None and scenario_mode != mode:
            continue
        if threshold is not None and scenario_threshold != f"{threshold:.2f}":
            continue
        selected.append(scenario)
    return selected


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    random.seed(config["seed"])

    if (args.start_phase is None) ^ (args.end_phase is None):
        raise SystemExit("Provide both --start_phase and --end_phase together.")
    if args.start_phase is not None and args.end_phase is not None:
        if args.start_phase > args.end_phase:
            raise SystemExit("--start_phase must be less than or equal to --end_phase.")
        phase_numbers = range(args.start_phase, args.end_phase + 1)
        phases = [PHASE_NUMBER_TO_ID[number] for number in phase_numbers]
        phase_label = f"{args.start_phase}-{args.end_phase}"
    else:
        phases = PHASE_GROUPS[args.phase]
        phase_label = args.phase

    scenarios = select_scenarios(config, args.mode, args.threshold)
    if not scenarios:
        raise SystemExit("No scenarios matched the requested --mode/--threshold filter.")

    print(f"Running phase selection: {phase_label}")
    print(f"Scenarios: {', '.join(scenarios)}")
    print(f"Force: {args.force}")

    for scenario in scenarios:
        for phase_id in phases:
            run_phase(config, phase_id, scenario, force=args.force)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
