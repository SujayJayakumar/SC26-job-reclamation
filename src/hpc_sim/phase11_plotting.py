from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import resolve_repo_path


TITLE_STYLE = {"fontweight": "bold", "fontsize": 16}
SUBPLOT_TITLE_STYLE = {"fontweight": "bold", "fontsize": 13}


def _load_phase_payload(phase_dir: Path) -> dict:
    payload_path = phase_dir / "payload.json"
    with payload_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _scenario_rows(config: dict, results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    weekly_rows: list[pd.DataFrame] = []

    for baseline in config["baselines"]:
        for threshold in config["thresholds"]:
            scenario = f"{baseline}_threshold_{threshold:.2f}"
            phase_dir = results_root / scenario / "phase_10_metrics"
            payload = _load_phase_payload(phase_dir)
            summary_rows.append(payload)

            weekly_path = phase_dir / "metrics_weekly.parquet"
            weekly = pd.read_parquet(weekly_path)
            if not weekly.empty:
                weekly = weekly.copy()
                weekly["scenario"] = scenario
                weekly["mode"] = payload["mode"]
                weekly["threshold"] = float(payload["threshold"])
                weekly_rows.append(weekly)

    summary = pd.DataFrame(summary_rows)
    summary["threshold"] = pd.to_numeric(summary["threshold"], errors="coerce")
    summary["mode"] = summary["mode"].astype("string")
    summary["scenario"] = summary["scenario"].astype("string")
    summary = summary.sort_values(["threshold", "mode"]).reset_index(drop=True)

    weekly_all = pd.concat(weekly_rows, ignore_index=True) if weekly_rows else pd.DataFrame()
    if not weekly_all.empty:
        weekly_all["week_start"] = pd.to_datetime(weekly_all["week_start"])
        weekly_all["threshold"] = pd.to_numeric(weekly_all["threshold"], errors="coerce")
    return summary, weekly_all


def _week_start(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce")
    if getattr(timestamps.dt, "tz", None) is not None:
        timestamps = timestamps.dt.tz_convert(None)
    return timestamps.dt.to_period("W-SUN").dt.start_time


def _load_matched_utilization_rows(config: dict, results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for threshold in config["thresholds"]:
        baseline_path = results_root / f"original_threshold_{threshold:.2f}" / "phase_09_simulation_loop" / "simulation_log.parquet"
        baseline = pd.read_parquet(baseline_path, columns=["timestamp", "node_id", "cpu_util", "total_util"])
        baseline["timestamp"] = pd.to_datetime(baseline["timestamp"])

        threshold_union: list[pd.DataFrame] = []
        for mode in ("aggressive", "buffered"):
            scenario = f"{mode}_threshold_{threshold:.2f}"
            scenario_path = results_root / scenario / "phase_09_simulation_loop" / "simulation_log.parquet"
            sim = pd.read_parquet(scenario_path, columns=["timestamp", "node_id", "cpu_util", "total_util"])
            sim["timestamp"] = pd.to_datetime(sim["timestamp"])
            matched = sim.merge(
                baseline.rename(columns={"cpu_util": "baseline_cpu_util", "total_util": "baseline_total_util"}),
                on=["timestamp", "node_id"],
                how="left",
            )
            rows.append(
                {
                    "threshold": float(threshold),
                    "mode": mode,
                    "scenario": scenario,
                    "matched_rows": int(len(matched)),
                    "baseline_cpu_util_matched": float(matched["baseline_cpu_util"].fillna(0.0).mean()),
                    "scenario_primary_cpu_util": float(matched["cpu_util"].fillna(0.0).mean()),
                    "scenario_total_cpu_util": float(matched["total_util"].fillna(0.0).mean()),
                }
            )
            threshold_union.append(sim[["timestamp", "node_id"]])

        if threshold_union:
            touched = pd.concat(threshold_union, ignore_index=True).drop_duplicates()
            baseline_union = touched.merge(baseline, on=["timestamp", "node_id"], how="left")
            rows.append(
                {
                    "threshold": float(threshold),
                    "mode": "original",
                    "scenario": f"original_threshold_{threshold:.2f}",
                    "matched_rows": int(len(baseline_union)),
                    "baseline_cpu_util_matched": float(baseline_union["cpu_util"].fillna(0.0).mean()),
                    "scenario_primary_cpu_util": float(baseline_union["cpu_util"].fillna(0.0).mean()),
                    "scenario_total_cpu_util": float(baseline_union["total_util"].fillna(0.0).mean()),
                }
            )

    return pd.DataFrame(rows).sort_values(["threshold", "mode"]).reset_index(drop=True)


def _load_weekly_primary_job_counts(config: dict) -> pd.DataFrame:
    phase_02_summary_path = resolve_repo_path(config, "Data/processed/phase_02_summary.json")
    with phase_02_summary_path.open("r", encoding="utf-8") as handle:
        phase_02_summary = json.load(handle)

    study_start = pd.Timestamp(phase_02_summary["common_overlap_window"]["start"], tz="UTC")
    study_end = pd.Timestamp(phase_02_summary["common_overlap_window"]["end"], tz="UTC")
    job_raw_path = resolve_repo_path(config, "Data/processed/job_raw.parquet")
    jobs = pd.read_parquet(job_raw_path, columns=["job_id", "job_latest_ts"])
    jobs["job_latest_ts"] = pd.to_datetime(jobs["job_latest_ts"], errors="coerce", utc=True)
    jobs = jobs.loc[jobs["job_latest_ts"].between(study_start, study_end)].copy()
    jobs["week_start"] = _week_start(jobs["job_latest_ts"])
    weekly = (
        jobs.groupby("week_start", as_index=False)
        .agg(primary_completed_jobs=("job_id", "nunique"))
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    return weekly


def _augment_weekly_with_job_uplift(weekly: pd.DataFrame, primary_weekly: pd.DataFrame) -> pd.DataFrame:
    if weekly.empty:
        return weekly

    merged = weekly.merge(primary_weekly, on="week_start", how="left")
    merged["primary_completed_jobs"] = pd.to_numeric(merged["primary_completed_jobs"], errors="coerce").fillna(0).astype("int64")
    merged["additional_jobs_served"] = pd.to_numeric(merged["completion_count"], errors="coerce").fillna(0).astype("int64")
    baseline = pd.to_numeric(merged["primary_completed_jobs"], errors="coerce").fillna(0.0)
    delta = pd.to_numeric(merged["additional_jobs_served"], errors="coerce").fillna(0.0)
    merged["job_service_increase_pct"] = delta.where(baseline > 0.0, 0.0).div(baseline.where(baseline > 0.0, 1.0)) * 100.0
    return merged


def _mode_color(mode: str) -> str:
    return {
        "original": "#6b7280",
        "buffered": "#1f77b4",
        "aggressive": "#d62728",
    }.get(mode, "#374151")


def _scenario_style(mode: str, threshold: float) -> dict[str, object]:
    key = (mode, round(float(threshold), 2))
    styles: dict[tuple[str, float], dict[str, object]] = {
        ("aggressive", 0.15): {"color": "#d62728", "linestyle": "-", "marker": "o"},
        ("aggressive", 0.20): {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},
        ("aggressive", 0.25): {"color": "#e377c2", "linestyle": ":", "marker": "^"},
        ("buffered", 0.15): {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
        ("buffered", 0.20): {"color": "#2ca02c", "linestyle": "--", "marker": "s"},
        ("buffered", 0.25): {"color": "#9467bd", "linestyle": ":", "marker": "^"},
        ("original", 0.15): {"color": "#6b7280", "linestyle": "-", "marker": "o"},
        ("original", 0.20): {"color": "#9ca3af", "linestyle": "--", "marker": "s"},
        ("original", 0.25): {"color": "#374151", "linestyle": ":", "marker": "^"},
    }
    return styles.get(key, {"color": _mode_color(mode), "linestyle": "-", "marker": "o"})


def _compact_number(value: float) -> str:
    absolute = abs(float(value))
    if absolute >= 1_000_000:
        return f"{value / 1_000_000.0:.1f}M"
    if absolute >= 1_000:
        return f"{value / 1_000.0:.1f}k"
    if absolute >= 100:
        return f"{value:.0f}"
    if absolute >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _add_bar_labels(ax: plt.Axes, decimals: bool = False) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        label = _compact_number(height) if decimals else f"{height:.0f}"
        x = patch.get_x() + patch.get_width() / 2.0
        offset = max(abs(height) * 0.01, 0.02)
        ax.text(x, height + offset, label, ha="center", va="bottom", fontsize=9, fontweight="bold")


def _add_precise_bar_labels(ax: plt.Axes, places: int) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        label = f"{height:.{places}f}"
        x = patch.get_x() + patch.get_width() / 2.0
        offset = max(abs(height) * 0.01, 0.02)
        ax.text(x, height + offset, label, ha="center", va="bottom", fontsize=9, fontweight="bold")


def _add_integer_bar_labels(ax: plt.Axes) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        label = f"{int(round(height)):,}"
        x = patch.get_x() + patch.get_width() / 2.0
        offset = max(abs(height) * 0.01, 0.02)
        ax.text(x, height + offset, label, ha="center", va="bottom", fontsize=9, fontweight="bold")


def _save_figure(fig: plt.Figure, destinations: list[Path]) -> None:
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_utilization_vs_baseline(matched_util: pd.DataFrame, destinations: list[Path]) -> None:
    plot_df = matched_util.loc[matched_util["mode"] != "original"].copy()
    plot_df["label"] = plot_df["mode"].str.title() + " " + plot_df["threshold"].map(lambda v: f"{v:.2f}")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = list(range(len(plot_df)))
    ax.bar(
        [i - 0.18 for i in x],
        plot_df["baseline_cpu_util_matched"],
        width=0.36,
        color="#a7c4bc",
        label="Matched Baseline CPU Util",
    )
    ax.bar(
        [i + 0.18 for i in x],
        plot_df["scenario_total_cpu_util"],
        width=0.36,
        color="#1f77b4",
        label="Total CPU Util With Opportunistic Jobs",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=25, ha="right")
    ax.set_ylabel("Mean CPU Utilization")
    ax.set_title("Utilization vs Matched Baseline", **TITLE_STYLE)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _add_bar_labels(ax, decimals=True)
    ax.margins(y=0.18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_figure(fig, destinations)


def _plot_resource_views(summary: pd.DataFrame, destinations: list[Path]) -> None:
    plot_df = summary.loc[summary["mode"] != "original"].copy()
    plot_df["label"] = plot_df["mode"].str.title() + " " + plot_df["threshold"].map(lambda v: f"{v:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(17.6, 5.8))
    colors = [_mode_color(mode) for mode in plot_df["mode"]]

    axes[0].bar(plot_df["label"], plot_df["throughput_proxy_cpu_core_minutes"] / 1_000_000.0, color=colors)
    axes[0].set_title("CPU View", **SUBPLOT_TITLE_STYLE)
    axes[0].set_ylabel("CPU Core-Minutes (Millions)")
    axes[0].grid(axis="y", alpha=0.25)
    _add_precise_bar_labels(axes[0], places=3)

    axes[1].bar(plot_df["label"], plot_df["throughput_proxy_gpu_device_minutes"], color=colors)
    axes[1].set_title("GPU View", **SUBPLOT_TITLE_STYLE)
    axes[1].set_ylabel("GPU Device-Minutes")
    axes[1].grid(axis="y", alpha=0.25)
    _add_bar_labels(axes[1], decimals=True)

    axes[2].bar(plot_df["label"], plot_df["cluster_utilization_improvement_pct"], color=colors)
    axes[2].set_title("Cluster View", **SUBPLOT_TITLE_STYLE)
    axes[2].set_ylabel("Cluster Utilization Improvement (%)")
    axes[2].grid(axis="y", alpha=0.25)
    _add_precise_bar_labels(axes[2], places=3)
    axes[2].margins(y=0.28)

    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout(w_pad=3.4, rect=[0, 0, 1, 0.98])
    _save_figure(fig, destinations)


def _plot_preemptions(summary: pd.DataFrame, destinations: list[Path]) -> None:
    plot_df = summary.loc[summary["mode"] != "original"].copy()
    plot_df["label"] = plot_df["mode"].str.title() + " " + plot_df["threshold"].map(lambda v: f"{v:.2f}")

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = [_scenario_style(mode, threshold)["color"] for mode, threshold in zip(plot_df["mode"], plot_df["threshold"])]
    ax.bar(plot_df["label"], plot_df["preemption_count"], color=colors)
    ax.set_ylabel("Preemption Count")
    ax.set_title("Preemptions by Scenario", **TITLE_STYLE)
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    _add_bar_labels(ax, decimals=True)
    _save_figure(fig, destinations)


def _plot_interference(weekly: pd.DataFrame, destinations: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    weekly_plot = weekly.loc[weekly["mode"] != "original"].copy()
    for scenario, frame in weekly_plot.groupby("scenario", sort=False):
        mode = str(frame["mode"].iloc[0])
        threshold = float(frame["threshold"].iloc[0])
        style = _scenario_style(mode, threshold)
        ax.plot(
            frame["week_start"],
            frame["additional_jobs_served"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=4.2,
            label=f"{mode.title()} {threshold:.2f}",
            color=style["color"],
            linestyle=style["linestyle"],
            alpha=0.95,
        )
    ax.set_title("Weekly Additional Jobs Served", **TITLE_STYLE)
    ax.set_ylabel("Additional Completed Jobs", fontsize=13, fontweight="bold")
    ax.set_xlabel("Week", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.tick_params(axis="both", labelsize=11)
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    ax.legend(frameon=False, ncol=3, prop={"weight": "bold", "size": 10})
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_figure(fig, destinations)


def _plot_threshold_comparison(summary: pd.DataFrame, destinations: list[Path]) -> None:
    plot_df = summary.loc[summary["mode"] != "original"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    for mode in ["buffered", "aggressive"]:
        frame = plot_df.loc[plot_df["mode"] == mode].sort_values("threshold")
        axes[0].plot(
            frame["threshold"],
            frame["completion_count"],
            marker="o",
            linewidth=2.2,
            color=_mode_color(mode),
            label=mode.title(),
        )
        axes[1].plot(
            frame["threshold"],
            frame["throughput_proxy_cpu_core_minutes"] / 1_000_000.0,
            marker="o",
            linewidth=2.2,
            color=_mode_color(mode),
            label=mode.title(),
        )

    axes[0].set_title("Completions vs Threshold", **SUBPLOT_TITLE_STYLE)
    axes[0].set_ylabel("Completed Opportunistic Jobs")
    axes[1].set_title("CPU Throughput vs Threshold", **SUBPLOT_TITLE_STYLE)
    axes[1].set_ylabel("CPU Core-Minutes (Millions)")
    for ax in axes:
        ax.set_xlabel("Threshold")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    _save_figure(fig, destinations)


def _plot_threshold_sensitivity(summary: pd.DataFrame, destinations: list[Path]) -> None:
    plot_df = summary.loc[summary["mode"] != "original"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))

    for mode in ["aggressive", "buffered"]:
        frame = plot_df.loc[plot_df["mode"] == mode].sort_values("threshold")
        style = _scenario_style(mode, 0.20)
        axes[0].plot(
            frame["threshold"],
            frame["mean_total_cpu_util"],
            marker=style["marker"],
            linewidth=2.4,
            linestyle=style["linestyle"],
            color=style["color"],
            label=mode.title(),
        )
        axes[1].plot(
            frame["threshold"],
            frame["preemption_count"],
            marker=style["marker"],
            linewidth=2.4,
            linestyle=style["linestyle"],
            color=style["color"],
            label=mode.title(),
        )

    axes[0].set_title("CPU Utilization vs Threshold", **SUBPLOT_TITLE_STYLE)
    axes[0].set_ylabel("Mean Total CPU Utilization", fontsize=12, fontweight="bold")
    axes[1].set_title("Preemptions vs Threshold", **SUBPLOT_TITLE_STYLE)
    axes[1].set_ylabel("Preemption Count", fontsize=12, fontweight="bold")

    for ax in axes:
        ax.set_xlabel("Threshold", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.tick_params(axis="both", labelsize=11)
        plt.setp(ax.get_xticklabels(), fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontweight="bold")
        ax.legend(frameon=False, prop={"weight": "bold", "size": 10})

    fig.suptitle("Threshold Sensitivity: Utilization and Preemption", **TITLE_STYLE)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_figure(fig, destinations)


def _plot_throughput_cpu_gpu(summary: pd.DataFrame, destinations: list[Path]) -> None:
    plot_df = summary.loc[summary["mode"] != "original"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))
    for mode in ["aggressive", "buffered"]:
        frame = plot_df.loc[plot_df["mode"] == mode].sort_values("threshold")
        style = _scenario_style(mode, 0.20)
        axes[0].plot(
            frame["threshold"],
            frame["throughput_proxy_cpu_core_minutes"] / 1_000_000.0,
            marker=style["marker"],
            linewidth=2.4,
            linestyle=style["linestyle"],
            color=style["color"],
            label=mode.title(),
        )
        axes[1].plot(
            frame["threshold"],
            frame["throughput_proxy_gpu_device_minutes"],
            marker=style["marker"],
            linewidth=2.4,
            linestyle=style["linestyle"],
            color=style["color"],
            label=mode.title(),
        )
    axes[0].set_title("CPU Throughput Proxy", **SUBPLOT_TITLE_STYLE)
    axes[0].set_ylabel("CPU Core-Minutes (Millions)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].set_title("GPU Throughput Proxy", **SUBPLOT_TITLE_STYLE)
    axes[1].set_ylabel("GPU Device-Minutes")
    axes[1].grid(axis="y", alpha=0.25)
    for ax in axes:
        ax.set_xlabel("Threshold")
        ax.legend(frameon=False)
    _save_figure(fig, destinations)


def _plot_completions_and_unique_jobs(summary: pd.DataFrame, destinations: list[Path]) -> None:
    plot_df = summary.loc[summary["mode"] != "original"].copy()
    plot_df["label"] = plot_df["mode"].str.title() + " " + plot_df["threshold"].map(lambda v: f"{v:.2f}")
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    x = list(range(len(plot_df)))
    ax.bar([i - 0.18 for i in x], plot_df["completion_count"], width=0.36, color="#2a9d8f", label="Completions")
    ax.bar([i + 0.18 for i in x], plot_df["unique_opportunistic_jobs_run"], width=0.36, color="#e9c46a", label="Unique Jobs Run")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=25, ha="right")
    ax.set_title("Completed Jobs and Unique Jobs Run", **TITLE_STYLE)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _add_integer_bar_labels(ax)
    ax.margins(y=0.18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_figure(fig, destinations)


def _plot_weekly_cpu_throughput(weekly: pd.DataFrame, destinations: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    weekly_plot = weekly.loc[weekly["mode"] != "original"].copy()
    for scenario, frame in weekly_plot.groupby("scenario", sort=False):
        mode = str(frame["mode"].iloc[0])
        threshold = float(frame["threshold"].iloc[0])
        style = _scenario_style(mode, threshold)
        ax.plot(
            frame["week_start"],
            frame["opportunistic_cpu_core_minutes"] / 1_000_000.0,
            marker=style["marker"],
            linewidth=1.8,
            label=f"{mode.title()} {threshold:.2f}",
            color=style["color"],
            linestyle=style["linestyle"],
            alpha=0.95,
        )
    ax.set_title("Weekly CPU Throughput", **TITLE_STYLE)
    ax.set_ylabel("CPU Core-Minutes (Millions)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    _save_figure(fig, destinations)


def _plot_weekly_completions(weekly: pd.DataFrame, destinations: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    weekly_plot = weekly.loc[weekly["mode"] != "original"].copy()
    for scenario, frame in weekly_plot.groupby("scenario", sort=False):
        mode = str(frame["mode"].iloc[0])
        threshold = float(frame["threshold"].iloc[0])
        style = _scenario_style(mode, threshold)
        ax.plot(
            frame["week_start"],
            frame["completion_count"],
            marker=style["marker"],
            linewidth=1.8,
            label=f"{mode.title()} {threshold:.2f}",
            color=style["color"],
            linestyle=style["linestyle"],
            alpha=0.95,
        )
    ax.set_title("Weekly Completed Opportunistic Jobs", **TITLE_STYLE)
    ax.set_ylabel("Completed Jobs")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    _save_figure(fig, destinations)


def _plot_weekly_gpu_throughput(weekly: pd.DataFrame, destinations: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    weekly_plot = weekly.loc[weekly["mode"] != "original"].copy()
    for scenario, frame in weekly_plot.groupby("scenario", sort=False):
        mode = str(frame["mode"].iloc[0])
        threshold = float(frame["threshold"].iloc[0])
        style = _scenario_style(mode, threshold)
        ax.plot(
            frame["week_start"],
            frame["opportunistic_gpu_device_minutes"],
            marker=style["marker"],
            linewidth=1.8,
            label=f"{mode.title()} {threshold:.2f}",
            color=style["color"],
            linestyle=style["linestyle"],
            alpha=0.95,
        )
    ax.set_title("Weekly GPU Throughput", **TITLE_STYLE)
    ax.set_ylabel("GPU Device-Minutes")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    _save_figure(fig, destinations)


def _plot_threshold_utilization_vs_baseline(matched_util: pd.DataFrame, threshold: float, destinations: list[Path]) -> None:
    frame = matched_util.loc[(matched_util["threshold"] == threshold) & (matched_util["mode"] != "original")].sort_values("mode").copy()
    frame["label"] = frame["mode"].str.title()
    x = range(len(frame))

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.bar([i - 0.18 for i in x], frame["baseline_cpu_util_matched"], width=0.36, color="#b8d8d8", label="Matched Baseline CPU Util")
    ax.bar([i + 0.18 for i in x], frame["scenario_total_cpu_util"], width=0.36, color="#1f77b4", label="Scenario Total CPU Util")
    ax.set_xticks(list(x))
    ax.set_xticklabels(frame["label"])
    ax.set_ylabel("Mean CPU Utilization")
    ax.set_title(f"Threshold {threshold:.2f}: Utilization vs Matched Baseline", **TITLE_STYLE)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _add_bar_labels(ax, decimals=True)
    ax.margins(y=0.18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_figure(fig, destinations)


def _plot_merged_threshold_preemptions(summary: pd.DataFrame, destinations: list[Path]) -> None:
    original = summary.loc[summary["mode"] == "original"].sort_values("threshold").head(1).copy()
    original["label"] = "Original"
    non_baseline = summary.loc[summary["mode"] != "original"].copy()
    non_baseline["label"] = non_baseline["mode"].str.title() + " " + non_baseline["threshold"].map(lambda v: f"{v:.2f}")
    frame = pd.concat([original, non_baseline.sort_values(["threshold", "mode"])], ignore_index=True)
    colors = [
        "#6b7280" if mode == "original" else _scenario_style(mode, threshold)["color"]
        for mode, threshold in zip(frame["mode"], frame["threshold"])
    ]

    fig, ax = plt.subplots(figsize=(10.8, 5.5))
    ax.bar(frame["label"], frame["preemption_count"], color=colors)
    ax.set_ylabel("Preemption Count", fontsize=13, fontweight="bold")
    ax.set_xlabel("Scenario / Threshold", fontsize=13, fontweight="bold")
    ax.set_title("Preemptions Across Thresholds", **TITLE_STYLE)
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=11)
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    _add_bar_labels(ax, decimals=True)
    ax.margins(y=0.16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_figure(fig, destinations)


def _plot_threshold_interference(weekly: pd.DataFrame, threshold: float, destinations: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 5.5))
    frame = weekly.loc[weekly["threshold"] == threshold].copy()
    if frame.empty:
        ax.set_title(f"Threshold {threshold:.2f}: Weekly Job-Service Increase", **TITLE_STYLE)
        ax.text(0.5, 0.5, "No weekly data", ha="center", va="center")
        _save_figure(fig, destinations)
        return

    for scenario, group in frame.groupby("scenario", sort=False):
        mode = str(group["mode"].iloc[0])
        style = _scenario_style(mode, threshold)
        ax.plot(
            group["week_start"],
            group["job_service_increase_pct"],
            linewidth=1.8,
            marker=style["marker"],
            markersize=3.2,
            label=mode.title(),
            color=style["color"],
            linestyle=style["linestyle"],
        )
    ax.set_title(f"Threshold {threshold:.2f}: Weekly Job-Service Increase", **TITLE_STYLE)
    ax.set_ylabel("Increase vs Baseline Completed Jobs (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, destinations)


def _plot_threshold_scenario_comparison(summary: pd.DataFrame, threshold: float, destinations: list[Path]) -> None:
    frame = summary.loc[summary["threshold"] == threshold].sort_values("mode").copy()
    frame["label"] = frame["mode"].str.title()
    colors = [_mode_color(mode) for mode in frame["mode"]]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4))
    axes = axes.ravel()

    axes[0].bar(frame["label"], frame["completion_count"], color=colors)
    axes[0].set_title("Completed Jobs", **SUBPLOT_TITLE_STYLE)
    axes[0].grid(axis="y", alpha=0.25)
    _add_bar_labels(axes[0], decimals=True)
    axes[0].margins(y=0.18)

    axes[1].bar(frame["label"], frame["throughput_proxy_cpu_core_minutes"] / 1_000_000.0, color=colors)
    axes[1].set_title("CPU Throughput", **SUBPLOT_TITLE_STYLE)
    axes[1].set_ylabel("CPU Core-Minutes (Millions)")
    axes[1].grid(axis="y", alpha=0.25)
    _add_bar_labels(axes[1], decimals=True)
    axes[1].margins(y=0.18)

    axes[2].bar(frame["label"], frame["throughput_proxy_gpu_device_minutes"], color=colors)
    axes[2].set_title("GPU Throughput", **SUBPLOT_TITLE_STYLE)
    axes[2].set_ylabel("GPU Device-Minutes")
    axes[2].grid(axis="y", alpha=0.25)
    _add_bar_labels(axes[2], decimals=True)
    axes[2].margins(y=0.18)

    axes[3].bar(frame["label"], frame["cluster_utilization_improvement_pct"], color=colors)
    axes[3].set_title("Cluster Utilization Improvement", **SUBPLOT_TITLE_STYLE)
    axes[3].set_ylabel("Improvement (%)")
    axes[3].grid(axis="y", alpha=0.25)
    _add_bar_labels(axes[3], decimals=True)
    axes[3].margins(y=0.18)

    fig.suptitle(f"Threshold {threshold:.2f}: Scenario Comparison", y=0.98, **TITLE_STYLE)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_figure(fig, destinations)


def run_phase_11(config: dict, output_dir: Path, scenario: str) -> dict[str, object]:
    results_root = resolve_repo_path(config, config["results_dir"])
    paper_outputs_dir = resolve_repo_path(config, config["paper_outputs_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_outputs_dir.mkdir(parents=True, exist_ok=True)

    summary, weekly = _scenario_rows(config, results_root)
    matched_util = _load_matched_utilization_rows(config, results_root)
    primary_weekly = _load_weekly_primary_job_counts(config)
    weekly = _augment_weekly_with_job_uplift(weekly, primary_weekly)

    figure_names = [
        "fig1_utilization_vs_baseline.png",
        "fig2_preemptions.png",
        "fig3_interference.png",
        "fig4_threshold_comparison.png",
        "fig5_throughput_cpu_gpu.png",
        "fig6_completions_and_unique_jobs.png",
        "fig7_weekly_cpu_throughput.png",
        "fig8_weekly_completions.png",
        "fig9_resource_views.png",
        "fig10_weekly_gpu_throughput.png",
        "fig15_threshold_sensitivity_util_vs_preemption.png",
    ]
    for threshold in config["thresholds"]:
        suffix = f"{threshold:.2f}".replace(".", "")
        figure_names.extend(
            [
                f"fig11_threshold_{suffix}_utilization_vs_baseline.png",
                f"fig12_threshold_{suffix}_preemptions.png",
                f"fig13_threshold_{suffix}_interference.png",
                f"fig14_threshold_{suffix}_scenario_comparison.png",
            ]
        )
    figure_paths = {name: str(output_dir / name) for name in figure_names}
    paper_paths = {name: str(paper_outputs_dir / name) for name in figure_names}

    _plot_utilization_vs_baseline(
        matched_util,
        [output_dir / "fig1_utilization_vs_baseline.png", paper_outputs_dir / "fig1_utilization_vs_baseline.png"],
    )
    _plot_preemptions(
        summary,
        [output_dir / "fig2_preemptions.png", paper_outputs_dir / "fig2_preemptions.png"],
    )
    _plot_interference(
        weekly,
        [output_dir / "fig3_interference.png", paper_outputs_dir / "fig3_interference.png"],
    )
    _plot_threshold_comparison(
        summary,
        [output_dir / "fig4_threshold_comparison.png", paper_outputs_dir / "fig4_threshold_comparison.png"],
    )
    _plot_throughput_cpu_gpu(
        summary,
        [output_dir / "fig5_throughput_cpu_gpu.png", paper_outputs_dir / "fig5_throughput_cpu_gpu.png"],
    )
    _plot_completions_and_unique_jobs(
        summary,
        [output_dir / "fig6_completions_and_unique_jobs.png", paper_outputs_dir / "fig6_completions_and_unique_jobs.png"],
    )
    _plot_weekly_cpu_throughput(
        weekly,
        [output_dir / "fig7_weekly_cpu_throughput.png", paper_outputs_dir / "fig7_weekly_cpu_throughput.png"],
    )
    _plot_weekly_completions(
        weekly,
        [output_dir / "fig8_weekly_completions.png", paper_outputs_dir / "fig8_weekly_completions.png"],
    )
    _plot_resource_views(
        summary,
        [output_dir / "fig9_resource_views.png", paper_outputs_dir / "fig9_resource_views.png"],
    )
    _plot_weekly_gpu_throughput(
        weekly,
        [output_dir / "fig10_weekly_gpu_throughput.png", paper_outputs_dir / "fig10_weekly_gpu_throughput.png"],
    )
    _plot_threshold_sensitivity(
        summary,
        [
            output_dir / "fig15_threshold_sensitivity_util_vs_preemption.png",
            paper_outputs_dir / "fig15_threshold_sensitivity_util_vs_preemption.png",
        ],
    )
    for threshold in config["thresholds"]:
        suffix = f"{threshold:.2f}".replace(".", "")
        _plot_threshold_utilization_vs_baseline(
            matched_util,
            float(threshold),
            [
                output_dir / f"fig11_threshold_{suffix}_utilization_vs_baseline.png",
                paper_outputs_dir / f"fig11_threshold_{suffix}_utilization_vs_baseline.png",
            ],
        )
        _plot_merged_threshold_preemptions(
            summary,
            [
                output_dir / f"fig12_threshold_{suffix}_preemptions.png",
                paper_outputs_dir / f"fig12_threshold_{suffix}_preemptions.png",
            ],
        )
        _plot_threshold_interference(
            weekly,
            float(threshold),
            [
                output_dir / f"fig13_threshold_{suffix}_interference.png",
                paper_outputs_dir / f"fig13_threshold_{suffix}_interference.png",
            ],
        )
        _plot_threshold_scenario_comparison(
            summary,
            float(threshold),
            [
                output_dir / f"fig14_threshold_{suffix}_scenario_comparison.png",
                paper_outputs_dir / f"fig14_threshold_{suffix}_scenario_comparison.png",
            ],
        )

    corrected_best = summary.loc[summary["mode"] != "original"].sort_values(
        ["completion_count", "throughput_proxy_cpu_core_minutes"],
        ascending=[False, False],
    ).iloc[0]

    summary_payload = {
        "status": "complete",
        "scenario": scenario,
        "plot_output_dir": str(output_dir),
        "paper_outputs_dir": str(paper_outputs_dir),
        "figures": figure_paths,
        "paper_output_figures": paper_paths,
        "best_scenario": str(corrected_best["scenario"]),
        "best_completion_count": int(corrected_best["completion_count"]),
        "best_cpu_core_minutes": float(corrected_best["throughput_proxy_cpu_core_minutes"]),
        "wait_time_plot_available": False,
        "wait_time_plot_reason": (
            "Primary-job queue wait-time reduction is not currently modeled end-to-end in the simulation, "
            "so a baseline-vs-reduced wait-time plot would be misleading."
        ),
        "weekly_jobs_served_plot_basis": (
            "Weekly additional jobs served is computed from opportunistic completion_count per week. "
            "Weekly job-service increase percent is computed against the baseline count of primary jobs whose "
            "job_latest_ts falls in the same study-window week."
        ),
        "recommended_optional_figures": [
            "fig5_throughput_cpu_gpu.png",
            "fig6_completions_and_unique_jobs.png",
            "fig7_weekly_cpu_throughput.png",
            "fig8_weekly_completions.png",
            "fig9_resource_views.png",
            "fig10_weekly_gpu_throughput.png",
            "fig11_threshold_015_utilization_vs_baseline.png",
            "fig11_threshold_020_utilization_vs_baseline.png",
            "fig11_threshold_025_utilization_vs_baseline.png",
            "fig14_threshold_015_scenario_comparison.png",
            "fig14_threshold_020_scenario_comparison.png",
            "fig14_threshold_025_scenario_comparison.png",
            "fig15_threshold_sensitivity_util_vs_preemption.png",
        ],
    }

    summary_path = output_dir / "plotting_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return summary_payload
