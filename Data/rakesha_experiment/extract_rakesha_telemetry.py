import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

TIME_FORMAT = "%a %b %d %H:%M:%S %Y"
CPU_FIELDS = [
    "cpu1_temp",
    "cpu2_temp",
    "cpu1_frequency",
    "cpu2_frequency",
    "cpu1_utilization",
    "cpu2_utilization",
]
GPU_FIELDS = [
    "power_consum",
    "utilization",
    "memory",
    "temperature",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Extract CPU and GPU telemetry for rakesha jobs.")
    parser.add_argument("--jobs", default="rakesha_jobs.json", help="Path to the extracted jobs JSON file.")
    parser.add_argument("--cpu", default="cpu_metrics.csv", help="Path to CPU telemetry CSV.")
    parser.add_argument("--gpu", default="gpu_status.csv", help="Path to GPU telemetry CSV.")
    parser.add_argument("--summary", default="rakesha_job_telemetry_summary.json", help="Output JSON summary path.")
    parser.add_argument("--cpu-samples", default="rakesha_job_cpu_samples.csv", help="Output CSV for matching CPU rows.")
    parser.add_argument("--gpu-samples", default="rakesha_job_gpu_samples.csv", help="Output CSV for matching GPU rows.")
    return parser.parse_args()


def parse_job_time(value):
    return datetime.strptime(value, TIME_FORMAT)


def parse_metric_time(value):
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def to_float(value):
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def gpu_card_to_node(card_id):
    parts = card_id.split("_")
    if len(parts) != 3:
        return None
    rack, node, _card = parts
    return f"r{rack}gn{node}"


def make_metric_bucket(field_names):
    bucket = {}
    for field in field_names:
        bucket[field] = {"sum": 0.0, "count": 0, "min": None, "max": None}
    return bucket


def update_metric_bucket(bucket, field_names, row):
    for field in field_names:
        value = to_float(row.get(field))
        if value is None:
            continue
        stats = bucket[field]
        stats["sum"] += value
        stats["count"] += 1
        stats["min"] = value if stats["min"] is None else min(stats["min"], value)
        stats["max"] = value if stats["max"] is None else max(stats["max"], value)


def finalize_metric_bucket(bucket):
    result = {}
    for field, stats in bucket.items():
        if stats["count"] == 0:
            result[field] = {"count": 0, "avg": None, "min": None, "max": None}
        else:
            result[field] = {
                "count": stats["count"],
                "avg": round(stats["sum"] / stats["count"], 4),
                "min": round(stats["min"], 4),
                "max": round(stats["max"], 4),
            }
    return result


def load_jobs(path):
    obj = json.loads(Path(path).read_text())
    jobs = []
    jobs_by_node = defaultdict(list)

    for job in obj["jobs"]:
        if not job.get("nodes") or not job.get("start_time") or not job.get("end_time"):
            continue
        start_time = parse_job_time(job["start_time"])
        end_time = parse_job_time(job["end_time"])
        if end_time < start_time:
            continue
        item = {
            "job_id": job["job_id"],
            "user": job.get("user"),
            "queue": job.get("queue"),
            "job_state": job.get("job_state"),
            "nodes": job["nodes"],
            "start_time": start_time,
            "end_time": end_time,
            "start_time_source": job.get("start_time_source"),
            "end_time_source": job.get("end_time_source"),
            "ncpus_req": job.get("ncpus_req"),
            "ncpus_used": job.get("ncpus_used"),
            "ngpus_req": job.get("ngpus_req"),
            "walltime_req": job.get("walltime_req"),
            "walltime_used": job.get("walltime_used"),
            "cpupercent_used": job.get("cpupercent_used"),
            "cpu_row_count": 0,
            "gpu_row_count": 0,
            "cpu_nodes_seen": set(),
            "gpu_nodes_seen": set(),
            "gpu_cards_seen": set(),
            "cpu_metrics": make_metric_bucket(CPU_FIELDS),
            "gpu_metrics": make_metric_bucket(GPU_FIELDS),
        }
        jobs.append(item)
        for node in item["nodes"]:
            jobs_by_node[node].append(item)

    for node_jobs in jobs_by_node.values():
        node_jobs.sort(key=lambda job: job["start_time"])

    return obj.get("summary", {}), jobs, jobs_by_node


def matching_jobs_for_row(node_jobs, timestamp):
    matches = []
    for job in node_jobs:
        if job["start_time"] <= timestamp <= job["end_time"]:
            matches.append(job)
        elif job["start_time"] > timestamp:
            break
    return matches


def stream_cpu(cpu_path, jobs_by_node, samples_path):
    matched_rows = 0
    with open(cpu_path, newline="") as source, open(samples_path, "w", newline="") as out:
        reader = csv.DictReader(source)
        fieldnames = [
            "job_id",
            "node_id",
            "timestamp",
            *CPU_FIELDS,
        ]
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            node_id = row["node_id"]
            node_jobs = jobs_by_node.get(node_id)
            if not node_jobs:
                continue
            timestamp = parse_metric_time(row["timestamp"])
            for job in matching_jobs_for_row(node_jobs, timestamp):
                job["cpu_row_count"] += 1
                job["cpu_nodes_seen"].add(node_id)
                update_metric_bucket(job["cpu_metrics"], CPU_FIELDS, row)
                writer.writerow({key: row.get(key) for key in fieldnames if key != "job_id"} | {"job_id": job["job_id"]})
                matched_rows += 1
    return matched_rows


def stream_gpu(gpu_path, jobs_by_node, samples_path):
    matched_rows = 0
    with open(gpu_path, newline="") as source, open(samples_path, "w", newline="") as out:
        reader = csv.DictReader(source)
        fieldnames = [
            "job_id",
            "node_id",
            "card_id",
            "timestamp",
            "is_healthy",
            *GPU_FIELDS,
        ]
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            node_id = gpu_card_to_node(row["card_id"])
            if not node_id:
                continue
            node_jobs = jobs_by_node.get(node_id)
            if not node_jobs:
                continue
            timestamp = parse_metric_time(row["timestamp"])
            for job in matching_jobs_for_row(node_jobs, timestamp):
                job["gpu_row_count"] += 1
                job["gpu_nodes_seen"].add(node_id)
                job["gpu_cards_seen"].add(row["card_id"])
                update_metric_bucket(job["gpu_metrics"], GPU_FIELDS, row)
                sample_row = {key: row.get(key) for key in fieldnames if key not in {"job_id", "node_id"}}
                sample_row["job_id"] = job["job_id"]
                sample_row["node_id"] = node_id
                writer.writerow(sample_row)
                matched_rows += 1
    return matched_rows


def build_output(job_summary, jobs, cpu_matches, gpu_matches):
    telemetry_jobs = []
    cpu_jobs = 0
    gpu_jobs = 0
    excluded_jobs = []

    for job in jobs:
        cpu_summary = finalize_metric_bucket(job["cpu_metrics"])
        gpu_summary = finalize_metric_bucket(job["gpu_metrics"])
        has_cpu = job["cpu_row_count"] > 0
        has_gpu = job["gpu_row_count"] > 0
        if has_cpu:
            cpu_jobs += 1
        if has_gpu:
            gpu_jobs += 1

        if not (has_cpu or has_gpu):
            excluded_jobs.append({
                "job_id": job["job_id"],
                "nodes": job["nodes"],
                "start_time": job["start_time"].strftime(TIME_FORMAT),
                "end_time": job["end_time"].strftime(TIME_FORMAT),
                "message": "No CPU or GPU values were found for this node and time range.",
            })
            continue

        telemetry_jobs.append({
            "job_id": job["job_id"],
            "user": job["user"],
            "queue": job["queue"],
            "job_state": job["job_state"],
            "nodes": job["nodes"],
            "start_time": job["start_time"].strftime(TIME_FORMAT),
            "end_time": job["end_time"].strftime(TIME_FORMAT),
            "start_time_source": job["start_time_source"],
            "end_time_source": job["end_time_source"],
            "ncpus_req": job["ncpus_req"],
            "ncpus_used": job["ncpus_used"],
            "ngpus_req": job["ngpus_req"],
            "walltime_req": job["walltime_req"],
            "walltime_used": job["walltime_used"],
            "cpupercent_used": job["cpupercent_used"],
            "cpu_row_count": job["cpu_row_count"],
            "gpu_row_count": job["gpu_row_count"],
            "cpu_nodes_seen": sorted(job["cpu_nodes_seen"]),
            "gpu_nodes_seen": sorted(job["gpu_nodes_seen"]),
            "gpu_cards_seen": sorted(job["gpu_cards_seen"]),
            "cpu_metrics": cpu_summary,
            "gpu_metrics": gpu_summary,
        })

    return {
        "job_summary": job_summary,
        "telemetry_summary": {
            "jobs_loaded": len(jobs),
            "jobs_with_any_telemetry": len(telemetry_jobs),
            "jobs_with_cpu_telemetry": cpu_jobs,
            "jobs_with_gpu_telemetry": gpu_jobs,
            "jobs_excluded_without_telemetry": len(excluded_jobs),
            "matched_cpu_rows": cpu_matches,
            "matched_gpu_rows": gpu_matches,
            "no_data_message": "Excluded jobs had no CPU or GPU values for their node and time range.",
        },
        "excluded_jobs_without_telemetry": excluded_jobs,
        "jobs": telemetry_jobs,
    }


def main():
    args = parse_args()
    job_summary, jobs, jobs_by_node = load_jobs(args.jobs)
    cpu_matches = stream_cpu(args.cpu, jobs_by_node, args.cpu_samples)
    gpu_matches = stream_gpu(args.gpu, jobs_by_node, args.gpu_samples)
    output = build_output(job_summary, jobs, cpu_matches, gpu_matches)
    Path(args.summary).write_text(json.dumps(output, indent=4))

    telemetry_summary = output["telemetry_summary"]
    print(f"Jobs loaded: {len(jobs)}")
    print(f"Jobs kept with telemetry: {telemetry_summary['jobs_with_any_telemetry']}")
    print(f"Jobs excluded without telemetry: {telemetry_summary['jobs_excluded_without_telemetry']}")
    if telemetry_summary["jobs_excluded_without_telemetry"]:
        print("No CPU or GPU values were found for the excluded jobs in their node/time range.")
    print(f"Matched CPU rows: {cpu_matches}")
    print(f"Matched GPU rows: {gpu_matches}")
    print(f"Summary file: {args.summary}")
    print(f"CPU samples file: {args.cpu_samples}")
    print(f"GPU samples file: {args.gpu_samples}")


if __name__ == "__main__":
    main()
