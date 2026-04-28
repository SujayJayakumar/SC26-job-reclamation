# Step 0 Architecture

This document locks the current execution model for the trace-driven HPC simulation pipeline and records the assumptions, methods, and technical details used by the implemented phases.

## System Goal

Build a deterministic, trace-driven HPC simulation system that reproduces:

`Monitoring -> Detection -> Reclamation -> Injection -> Preemption -> Metrics`

using real production HPC telemetry and job logs, with one-command execution, reproducible outputs, and phase-by-phase disk artifacts.

## Global Constraints

- Entry point: `bash scripts/run_all.sh`
- Single-phase entry point: `bash scripts/run_phase.sh <phase_id> --scenario <scenario>`
- Global seed: `42`
- Every phase reads from disk and writes to disk
- Every phase is independently rerunnable
- Intermediate outputs are preserved
- All scenarios share the same common study window
- Outputs must map directly to paper tables and figures

## Scenario Matrix

Scenarios are generated as:

- `original_threshold_0.15`
- `original_threshold_0.20`
- `original_threshold_0.25`
- `aggressive_threshold_0.15`
- `aggressive_threshold_0.20`
- `aggressive_threshold_0.25`
- `buffered_threshold_0.15`
- `buffered_threshold_0.20`
- `buffered_threshold_0.25`

The threshold names refer to the primary-job underutilization threshold used by detection.

## Common Study Window

The current pipeline restricts all simulation phases to the common overlap across:

- CPU telemetry
- GPU telemetry
- memory telemetry
- job logs

Locked overlap window:

- start: `2025-06-08 18:00:00`
- end: `2026-01-31 23:50:00`
- duration: about `237.24` days, roughly `7.8 months`

This window is enforced during preprocessing so later phases run on a solid, cross-source-correlated dataset.

## Data Contracts

Processed Phase 1 outputs:

- `Data/processed/cpu_raw.parquet`
- `Data/processed/gpu_raw.parquet`
- `Data/processed/memory_raw.parquet`
- `Data/processed/job_raw.parquet`

Processed Phase 2 outputs:

- `Data/processed/node_metrics.parquet`
- `Data/processed/job_mapped.parquet`
- `Data/processed/baseline_summary.json`

Scenario phase outputs are written to:

- `results/<scenario>/<phase_id>/manifest.json`
- `results/<scenario>/<phase_id>/payload.json`

## Locked Technical Assumptions

- Unified evaluation interval: `10 minutes`
- CPU, GPU, and memory utilization are normalized to fractions in `[0, 1]`
- CPU reclaimable capacity is inferred over `128` cores per node
- One node has `4` physical GPU cards, but multi-node jobs can produce summed logical GPU allocation on a node-time slice in the trace-derived overlay
- Memory is a safety guard, not a reclaimable resource
- Queue is assumed non-empty
- Opportunistic jobs are modeled as preemptible and checkpointable
- Opportunistic job durations use actual observed runtime when available, not only requested walltime
- GPU reclaimability is a simulation assumption based on compressibility of low-utilization multi-GPU workloads
- VRAM-aware enforcement and MIG-aware partitioning are future work
- Resource pooling across multiple nodes is not implemented yet

## Baseline Summary

The paper should treat `Data/processed/baseline_summary.json` as the baseline fact sheet derived from the current processed artifacts.

Study-window baseline:

- common study window: `2025-06-08 18:00:00` to `2026-01-31 23:50:00`
- duration: `237.24` days
- nodes: `422`
- aligned 10-minute timestamps: `34,164`

Whole-cluster baseline utilization means:

- CPU: `0.7008`
- GPU: `0.0053`
- memory: `0.0918`

Job-population summary over the overlapping study window:

- total jobs in raw trace: `210,287`
- jobs overlapping the study window: `101,099`
- unique users overlapping the study window: `160`
- average requested cores per job: `248.86`
- median requested cores per job: `128`
- p90 requested cores per job: `640`
- GPU-job fraction: `13.78%`
- average requested GPUs across all jobs: `0.2368`
- average requested GPUs among GPU jobs only: `1.7191`
- average observed runtime: `6.39` hours
- average requested runtime: `42.12` hours

Important interpretation:

- the `0.16-0.17` CPU values seen in the matched-baseline utilization plots are not the whole-cluster average
- they are matched baseline means computed only on the node-time rows touched by the non-baseline scenario
- for threshold `0.15`, those matched baseline CPU means are about:
  - aggressive: `0.1595`
  - buffered: `0.1602`
- the actual whole-cluster baseline CPU mean remains about `0.7008`

Limitation:

- average CPU or GPU utilization per job is not directly observable from the current trace because telemetry is node-based rather than job-attributed
- for writing, use cluster-level baseline means, matched-baseline means on touched rows, and job request/runtime summaries instead of claiming true per-job utilization

## Phase-by-Phase Methods

## Phase 1: Data Loading

Purpose:

- load raw source files
- standardize schemas
- normalize timestamps onto a common UTC-aware basis

Outputs:

- `cpu_raw.parquet`
- `gpu_raw.parquet`
- `memory_raw.parquet`
- `job_raw.parquet`
- `phase_01_summary.json`

## Phase 2: Preprocessing

Purpose:

- unify telemetry to `10-minute` intervals
- restrict to the common overlap window
- repair job temporal windows
- build explicit node-time continuity

Methods:

- CPU `5-minute -> 10-minute` mean aggregation
- GPU aggregation to node level as mean of card telemetry
- memory aggregation to node level
- job expansion from `job_id -> node_id`
- extraction of allocated cores and GPUs per node
- densification of node telemetry onto a complete node-time grid
- forward-fill support for later phases

Outputs:

- `node_metrics.parquet`
- `job_mapped.parquet`
- `phase_02_summary.json`

## Phase 3: State Construction

Purpose:

- build the canonical node-time state table used by simulation

State fields:

- `timestamp`
- `node_id`
- `cpu_util`
- `gpu_util`
- `memory_util`
- `allocated_cores`
- `allocated_gpus`
- `job_id`
- `active_job_count`
- `telemetry_present_any`

Methods:

- merge telemetry with mapped primary jobs
- normalize utilization to `[0, 1]`
- enforce full `timestamp x node` continuity
- forward-fill missing telemetry across explicit gaps
- maintain one canonical primary `job_id` for convenience while preserving summed allocations and active-job count

Output:

- `state_cache.parquet`

## Phase 4: Underutilization Detection

Purpose:

- identify node-time slices where reclaimable slack exists

CPU detection rule:

- current interval and previous interval both satisfy:
  - `cpu_util < threshold`
  - `memory_util < 0.50`
  - no memory spike where `current_memory_util > 2 * previous_memory_util`

GPU detection rule:

- current interval and previous interval both satisfy:
  - `allocated_gpus > 1`
  - `gpu_util < 0.80`
  - `memory_util < 0.50`
  - no memory spike where `current_memory_util > 2 * previous_memory_util`

Important:

- GPU reclaim detection is independent of CPU underutilization
- this is what allows GPU-idle but CPU-moderate nodes to be reclaimed

Output:

- `underutilized_nodes.parquet`

## Phase 5: Reclamation

Purpose:

- compute reclaimable CPU and GPU resources per eligible node-time slice

CPU reclamation:

- `reclaimable_cores = int(128 * (1 - cpu_util))`
- applied only when CPU underutilization is true

GPU reclamation:

- only considered when GPU underutilization is true
- low-utilization multi-GPU workloads are compacted so remaining cards stay at or below `80%` utilization
- reclaimed GPUs are the cards freed by that compaction

Output:

- `reclamation_plan.parquet`

## Phase 6: Opportunistic Job Modeling

Purpose:

- build the queue-side candidate space for opportunistic execution

Source:

- real job logs from the monitored trace

Methods:

- queue is ordered by `qtime`
- candidates are drawn from upcoming queue jobs
- runtime uses actual observed runtime when available:
  - `resources_used.walltime`
  - otherwise `etime - stime`
  - otherwise requested walltime
- jobs may be CPU-only, GPU-only, or mixed

Queue search policy:

- search up to `scan_limit = 10000` jobs deep in the waiting queue
- collect up to `selection_window = 25` feasible candidates
- choose `best_fit`

Output:

- `opportunistic_job_pool.parquet`

## Phase 7 and Phase 8

These earlier intermediate injection and preemption phases still exist on disk, but the main decision engine is now Phase 9.

They remain useful as diagnostics, but the authoritative simulation behavior comes from the stateful Phase 9 loop.

## Phase 9: Simulation Loop

Purpose:

- execute the stateful scenario simulation over the full timeline

Modes:

- `original`: no opportunistic work, raw baseline
- `aggressive`: primary-only reserve equals the threshold
- `buffered`: primary-only reserve equals the configured buffered reserve

Current primary reserve values:

- threshold `0.15` -> buffered cap `0.25`
- threshold `0.20` -> buffered cap `0.30`
- threshold `0.25` -> buffered cap `0.40`

Important interpretation:

- the threshold is the low-utilization trigger for admission
- the threshold/buffer reserve is reserved for the primary workload only
- opportunistic work is allowed to occupy the remaining reclaimable hardware
- aggressive mode reserves less primary slack, so it can admit more opportunistic work
- buffered mode reserves more primary slack, so it is usually more conservative but may preempt less

Stateful simulation methods:

- real waiting queue lifecycle
- one queue job cannot occupy multiple nodes at the same time
- jobs are removed from the queue when assigned
- preempted jobs are deferred and requeued later
- multiple opportunistic jobs per node are allowed
- current cap: `max_injected_jobs_per_node = 4`
- opportunistic admission is driven by reclaimable CPU and GPU resources, not by a total-utilization cap
- active opportunistic work may continue even after the node leaves the detection window
- preemption protects the primary workload whenever:
  - primary CPU utilization rises above its reserved threshold/buffer band, or
  - active opportunistic CPU/GPU allocations exceed the currently reclaimable envelope

Not implemented:

- multi-node pooled placement

Pooling status:

- Phase 9 only measures pooled-fit opportunity as a diagnostic
- it does not actually allocate one opportunistic job across multiple nodes

Resumability:

- Phase 9 now writes resumable checkpoint state under `results/<scenario>/phase_09_simulation_loop/_resume/`
- checkpoint contents include queue position, waiting queue, active jobs, progress counters, and remaining runtime overrides
- partial simulation log batches are flushed as parquet parts
- rerunning without `--force` can resume from the latest checkpoint
- rerunning with `--force` clears the checkpoint and starts fresh

Output:

- `simulation_log.parquet`
- `_resume/checkpoint_state.json`
- `_resume/batches/part-*.parquet`

## Phase 10: Metrics

Purpose:

- compute paper-facing comparison metrics

Current metrics:

- utilization improvement
- preemption count
- interference events
- throughput proxy
- completion count
- unique opportunistic jobs run

Throughput proxy:

- opportunistic CPU core-minutes
- opportunistic GPU device-minutes

Output:

- `metrics_summary.parquet`
- `metrics_weekly.parquet`
- `metrics_summary.json`

Throughput-gain fields:

- `relative_cpu_throughput_gain_fraction = opportunistic_cpu_core_minutes / baseline_primary_core_minutes`
- `relative_cpu_throughput_gain_pct = 100 * relative_cpu_throughput_gain_fraction`

## Phase 11: Plotting

Purpose:

- generate paper-facing figures from the corrected Phase 10 metrics tables

Core figures:

- `fig1_utilization_vs_baseline.png`
- `fig2_preemptions.png`
- `fig3_interference.png`
- `fig4_threshold_comparison.png`

Additional figures retained for selection:

- `fig5_throughput_cpu_gpu.png`
- `fig6_completions_and_unique_jobs.png`
- `fig7_weekly_cpu_throughput.png`
- `fig8_weekly_completions.png`
- `fig9_resource_views.png`
- `fig10_weekly_gpu_throughput.png`
- `fig11_threshold_015_utilization_vs_baseline.png`
- `fig11_threshold_020_utilization_vs_baseline.png`
- `fig11_threshold_025_utilization_vs_baseline.png`
- `fig12_threshold_015_preemptions.png`
- `fig12_threshold_020_preemptions.png`
- `fig12_threshold_025_preemptions.png`
- `fig13_threshold_015_interference.png`
- `fig13_threshold_020_interference.png`
- `fig13_threshold_025_interference.png`
- `fig14_threshold_015_scenario_comparison.png`
- `fig14_threshold_020_scenario_comparison.png`
- `fig14_threshold_025_scenario_comparison.png`

Heterogeneous-system coverage:

- CPU-only views are plotted separately
- GPU-only views are plotted separately
- cluster-level utilization improvement is plotted separately
- a combined resource-view figure is generated so CPU, GPU, and cluster-level outcomes can be compared side by side
- each threshold now also has its own baseline-inclusive comparison figures so `original`, `aggressive`, and `buffered` can be compared without cross-threshold crowding

Important limitation:

- weekly average wait-time reduction is not plotted because the current simulation does not propagate primary-job queue wait times end-to-end
- plotting a wait-time reduction figure from the current model would therefore be misleading
- baseline utilization figures use matched baseline rows from the `original` simulation log for the same `(timestamp, node_id)` slices touched by each non-baseline scenario, so the bars are comparable instead of mixing full-cluster baseline means with sparse simulation-event means
- the legacy `fig3_*interference*` filenames are retained for continuity, but they now emphasize weekly additional jobs served and weekly job-service increase over baseline rather than weekly interruption counts

## Final Corrected Threshold Report

The following numbers are the current validated results after correcting the Phase 9 policy so that:

- threshold and buffer are reserved for the primary workload only
- opportunistic work uses the remaining reclaimable hardware
- preemption happens only when primary growth or the current reclaimable envelope would be violated

Corrected Phase 9 and Phase 10 results:

- `buffered_threshold_0.15`
  - injections: `3792`
  - completions: `3383`
  - preemptions: `924`
  - unique opportunistic jobs: `5506`
  - opportunistic CPU throughput: `93,449,080` core-minutes
  - opportunistic GPU throughput: `202,970` device-minutes
  - cluster CPU utilization improvement: `0.5064%`
  - relative CPU throughput gain: `0.7226%`
- `aggressive_threshold_0.15`
  - injections: `4029`
  - completions: `3338`
  - preemptions: `1083`
  - unique opportunistic jobs: `5501`
  - opportunistic CPU throughput: `93,137,290` core-minutes
  - opportunistic GPU throughput: `196,860` device-minutes
  - cluster CPU utilization improvement: `0.5047%`
  - relative CPU throughput gain: `0.7202%`
- `buffered_threshold_0.20`
  - injections: `3778`
  - completions: `3408`
  - preemptions: `772`
  - unique opportunistic jobs: `5519`
  - opportunistic CPU throughput: `93,847,400` core-minutes
  - opportunistic GPU throughput: `219,170` device-minutes
  - cluster CPU utilization improvement: `0.5085%`
  - relative CPU throughput gain: `0.7257%`
- `aggressive_threshold_0.20`
  - injections: `3882`
  - completions: `3406`
  - preemptions: `805`
  - unique opportunistic jobs: `5519`
  - opportunistic CPU throughput: `93,762,760` core-minutes
  - opportunistic GPU throughput: `220,090` device-minutes
  - cluster CPU utilization improvement: `0.5081%`
  - relative CPU throughput gain: `0.7250%`
- `buffered_threshold_0.25`
  - injections: `3785`
  - completions: `3446`
  - preemptions: `633`
  - unique opportunistic jobs: `5519`
  - opportunistic CPU throughput: `93,956,540` core-minutes
  - opportunistic GPU throughput: `218,270` device-minutes
  - cluster CPU utilization improvement: `0.5091%`
  - relative CPU throughput gain: `0.7265%`
- `aggressive_threshold_0.25`
  - injections: `3801`
  - completions: `3397`
  - preemptions: `621`
  - unique opportunistic jobs: `5519`
  - opportunistic CPU throughput: `93,942,250` core-minutes
  - opportunistic GPU throughput: `218,840` device-minutes
  - cluster CPU utilization improvement: `0.5091%`
  - relative CPU throughput gain: `0.7264%`

Current strongest configuration:

- `buffered_threshold_0.25`

It gives the highest completion count and the highest CPU throughput of the corrected matrix, while keeping preemptions below the lower-threshold modes.

## Reproducibility CLI

Mandatory top-level entrypoint:

```bash
python main.py --phase preprocess
python main.py --phase simulate
python main.py --phase metrics
python main.py --phase plots
python main.py --mode buffered --threshold 0.20
```

Implemented behavior:

- `python main.py --phase preprocess`
  - runs Phases `01` to `06`
- `python main.py --phase simulate`
  - runs Phase `09`
- `python main.py --phase metrics`
  - runs Phase `10`
- `python main.py --phase plots`
  - runs Phase `11`
- `python main.py --mode buffered --threshold 0.20`
  - runs the full available pipeline for scenario `buffered_threshold_0.20`

CLI notes:

- the canonical script is [`main.py`](/home/sujay/Desktop/SUJAY/SC Paper/main.py)
- `--mode` accepts `original`, `aggressive`, `buffered`
- `--threshold` accepts `0.15`, `0.20`, `0.25`
- `--force` re-runs a phase even if its manifest already exists
- `--start_phase` and `--end_phase` support inclusive resumable phase ranges
  - example:
    - `python main.py --start_phase 5 --end_phase 9`
- for this repo's local environment, `.venv/bin/python main.py ...` is the validated invocation on this machine
- the CLI delegates to the existing phase handlers in [`pipeline.py`](/home/sujay/Desktop/SUJAY/SC Paper/src/hpc_sim/pipeline.py), so it remains consistent with the internal pipeline implementation
- the current grouped CLI intentionally skips intermediate diagnostic-only stages `07` and `08` in the top-level reproducibility path because the final stateful simulation logic is carried by Phase `09`
- phase resumability is manifest-based:
  - if a phase manifest already exists, it is skipped by default
  - `--force` re-runs it
  - Phase `09` additionally supports checkpoint-aware resume behavior internally

## Final Run Script

Validated final entrypoint:

```bash
bash scripts/run_all.sh
```

Current behavior:

- runs `preprocess`, `simulate`, `metrics`, and `plots` through [`main.py`](/home/sujay/Desktop/SUJAY/SC Paper/main.py)
- covers all `9` baseline-threshold experiments
- runs final validation checks
- exports strict AD/AE-friendly outputs into [`outputs/`](/home/sujay/Desktop/SUJAY/SC Paper/outputs)
- writes timestamped logs into [`outputs/logs`](/home/sujay/Desktop/SUJAY/SC Paper/outputs/logs)
- prefers the local `.venv` interpreter automatically on this machine

## Validation Checks

Implemented validation helper:

```bash
python scripts/validate_pipeline.py --config config/default.json
```

Generated artifacts:

- [`outputs/tables/validation_checks.csv`](/home/sujay/Desktop/SUJAY/SC Paper/outputs/tables/validation_checks.csv)
- [`outputs/logs/validation_summary.json`](/home/sujay/Desktop/SUJAY/SC Paper/outputs/logs/validation_summary.json)

Auto-checks now enforced:

- opportunistic CPU allocation stays within `128` cores per node row
- opportunistic job count per node stays within the configured final cap
  - final cap is `4`, not `1`, because the corrected simulator intentionally supports multi-job packing
- no negative opportunistic resources
- timestamps remain monotonic in Phase `09` logs
- preemption rows preserve valid non-negative post-preemption state within the configured job-count cap

Important final-model note:

- the original checklist item `max 1 opportunistic job/node` is **not** used in the final artifact checks because it conflicts with the corrected multi-job-per-node simulation policy
- GPU totals in the final logs are treated as logical allocated demand, so validation checks non-negativity and state consistency rather than a strict per-row `<= 4 GPU` physical bound

## Output Contract

Strict exported artifact contract:

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

Current implementation:

- tables are exported by [`scripts/export_outputs.py`](/home/sujay/Desktop/SUJAY/SC Paper/scripts/export_outputs.py)
- figures are copied from [`results/paper_outputs`](/home/sujay/Desktop/SUJAY/SC Paper/results/paper_outputs) into [`outputs/figures`](/home/sujay/Desktop/SUJAY/SC Paper/outputs/figures)
- logs are produced by [`scripts/run_all.sh`](/home/sujay/Desktop/SUJAY/SC Paper/scripts/run_all.sh)
- additional convenience exports currently included:
  - [`outputs/tables/completions.csv`](/home/sujay/Desktop/SUJAY/SC Paper/outputs/tables/completions.csv)
  - [`outputs/tables/validation_checks.csv`](/home/sujay/Desktop/SUJAY/SC Paper/outputs/tables/validation_checks.csv)

## Sanity Constraints

The repo now records these as reproducibility checks rather than hardcoded assumptions:

- `buffered < aggressive (interference)`
- `aggressive >= buffered >= original (utilization)`
- `preemptions increase with threshold`

Current observed status from [`outputs/logs/validation_summary.json`](/home/sujay/Desktop/SUJAY/SC Paper/outputs/logs/validation_summary.json):

- `buffered < aggressive (interference)`
  - true at thresholds `0.15` and `0.20`
  - false at `0.25`
- `aggressive >= buffered >= original (utilization)`
  - false for all three thresholds in the final corrected matrix
  - interpretation:
    - the final corrected policy makes aggressive and buffered very close, and original remains the baseline reference; this heuristic should not be written as a guaranteed invariant
- `preemptions increase with threshold`
  - false for both `aggressive` and `buffered` in the final corrected matrix
  - interpretation:
    - this is not a universal property of the corrected reserve-based policy and should be treated as a heuristic expectation rather than a required outcome

Artifact-sharing note:

- Phase 03 state construction is scenario-independent in the current design, so non-canonical scenario payloads reference the canonical `original_threshold_0.15` state cache through `shared_artifact_source`
- Phases 04 to 06 are threshold-dependent but mode-independent, so `aggressive_*` and `buffered_*` payloads for those phases reference the corresponding `original_threshold_*` artifacts through `shared_artifact_source`
- This keeps scenario-local payloads non-placeholder while making the shared-data contract explicit

## Broken vs Corrected Policy Comparison

The previous broken-policy matrix incorrectly treated the threshold/buffer as a cap on `primary + opportunistic` utilization. The corrected policy treats the threshold/buffer as a primary-only reserve.

Impact of the correction:

- `buffered_threshold_0.15`
  - old broken-policy: `2232` injections, `1361` completions, `963` preemptions
  - corrected policy: `3792` injections, `3383` completions, `924` preemptions
- `aggressive_threshold_0.15`
  - old broken-policy: `2244` injections, `1301` completions, `1104` preemptions
  - corrected policy: `4029` injections, `3338` completions, `1083` preemptions
- `buffered_threshold_0.20`
  - old broken-policy: `2488` injections, `1594` completions, `972` preemptions
  - corrected policy: `3778` injections, `3408` completions, `772` preemptions
- `aggressive_threshold_0.20`
  - old broken-policy: `2134` injections, `1324` completions, `937` preemptions
  - corrected policy: `3882` injections, `3406` completions, `805` preemptions
- `buffered_threshold_0.25`
  - old broken-policy: `3405` injections, `2032` completions, `1471` preemptions
  - corrected policy: `3785` injections, `3446` completions, `633` preemptions
- `aggressive_threshold_0.25`
  - old broken-policy: `2191` injections, `1382` completions, `902` preemptions
  - corrected policy: `3801` injections, `3397` completions, `621` preemptions

The corrected policy is therefore the final policy to use for reporting, and the earlier conservative matrix should be treated only as an intermediate debugging stage.

## Future Work Already Identified

- multi-node pooled placement
- stronger GPU packing policy
- VRAM-aware GPU reclaim modeling
- MIG-aware enforcement
- queue-policy sensitivity study beyond current best-fit settings
- additional checkpoint management and cleanup utilities
