# Observability-Driven Runtime Resource Reclamation for Heterogeneous HPC Systems

This repository contains the trace-driven simulation framework, evaluation pipeline, and plotting code used for the SC26 paper on observability-driven runtime resource reclamation for heterogeneous HPC systems.

The artifact implements a non-intrusive runtime augmentation layer that:
- detects reclaimable CPU and GPU capacity from node-level telemetry
- injects opportunistic checkpointable jobs
- applies a reserve-based preemption policy to protect primary workloads
- reproduces the reported metrics and figures from the paper

## Repository Layout

- `main.py`
  - top-level CLI entrypoint
- `src/hpc_sim/`
  - simulation pipeline implementation
- `scripts/`
  - helper scripts for execution, validation, output export, and baseline summary
- `config/default.json`
  - default experiment configuration
- `docs/STEP_0_ARCHITECTURE.md`
  - canonical architecture, pipeline, assumptions, metrics, and reproducibility notes
- `results/paper_outputs/`
  - generated paper figures
- `outputs/`
  - exported AD/AE-friendly figures, tables, and logs

## Dataset

The raw dataset is hosted on Zenodo:

- DOI: `https://doi.org/10.5281/zenodo.19842071`

After downloading the dataset archive, extract the files directly into the repository `Data/` directory so that the following required files exist:

- `Data/cpu_metrics.csv`
- `Data/gpu_status.csv`
- `Data/memory_data.csv`
- `Data/merged_all_jobs.jsonl`

The pipeline expects the dataset under `Data/`, not `dataset/`.

## Environment Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main CLI

Mandatory top-level commands:

```bash
python main.py --phase preprocess
python main.py --phase simulate
python main.py --phase metrics
python main.py --phase plots

python main.py --mode buffered --threshold 0.20
```

Supported options:

- `--mode {original,aggressive,buffered}`
- `--threshold {0.15,0.20,0.25}`
- `--force`
- `--start_phase N --end_phase M`

Example resumable range execution:

```bash
python main.py --start_phase 5 --end_phase 9
```

## Full Reproducibility Run

Run the full pipeline, validation, and export workflow:

```bash
bash scripts/run_all.sh
```

This performs:
- preprocessing
- simulation
- metrics generation
- plot generation
- validation checks
- strict output export into `outputs/`

## Outputs

The final exported artifact contract is:

```text
outputs/
    tables/
        utilization.csv
        preemptions.csv
        interference.csv

    figures/
        *.png

    logs/
        run_*.log
```

Additional validation and convenience outputs are also generated:
- `outputs/tables/completions.csv`
- `outputs/tables/validation_checks.csv`
- `outputs/logs/validation_summary.json`
- `outputs/export_manifest.json`

## Validation

You can run validation directly with:

```bash
python scripts/validate_pipeline.py --config config/default.json
```

Current checks include:
- opportunistic CPU allocation stays within 128 cores per node row
- opportunistic job count per node stays within the configured cap
- no negative opportunistic resources
- monotonic Phase 9 timestamps
- valid post-preemption state

Note: the final corrected simulator intentionally allows up to `max_injected_jobs_per_node = 4`, so the validation logic checks against the configured cap rather than a one-job-per-node rule.

## Final Reported Results

Best validated configuration:
- `buffered_threshold_0.25`

Key results:
- `0.5091%` cluster-wide utilization improvement
- `93,956,540` opportunistic CPU core-minutes recovered
- `218,270` opportunistic GPU device-minutes recovered
- `3,446` additional completed opportunistic jobs

The final corrected policy uses a primary-reserve interpretation:
- threshold and buffer are reserved for the primary workload
- opportunistic jobs use remaining reclaimable capacity
- preemption occurs when primary growth or reclaimable-envelope violation would affect safe execution

## Notes

- The artifact is deterministic under the provided configuration and fixed seed.
- The simulator is trace-driven and does not require live scheduler access.
- Weekly primary-job wait-time reduction is not modeled end-to-end and should not be inferred from the current outputs.

For the authoritative technical description, see:
- `docs/STEP_0_ARCHITECTURE.md`
