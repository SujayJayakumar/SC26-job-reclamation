import json
import re
from datetime import datetime
from pathlib import Path

TARGET_USER = "rakesha"
SOURCE_FILES = [
    "final_full_job_data.txt",
    "full_job_details.txt",
    "qstat_output.txt",
]
OUTPUT_FILE = "rakesha_jobs.json"
TIME_FORMAT = "%a %b %d %H:%M:%S %Y"
REQUIRED_TIME_FIELDS = ("qtime", "mtime")
START_TIME_CANDIDATES = ("stime", "etime", "qtime")

FIELD_PATTERNS = {
    "job_id": r"^Job Id:\s*(\S+)",
    "job_name": r"^\s*Job_Name\s*=\s*(.+)",
    "job_owner": r"^\s*Job_Owner\s*=\s*(\S+)",
    "euser": r"^\s*euser\s*=\s*(\S+)",
    "queue": r"^\s*queue\s*=\s*(\S+)",
    "job_state": r"^\s*job_state\s*=\s*(\S+)",
    "qtime": r"^\s*qtime\s*=\s*(.+)",
    "stime": r"^\s*stime\s*=\s*(.+)",
    "etime": r"^\s*etime\s*=\s*(.+)",
    "ctime": r"^\s*ctime\s*=\s*(.+)",
    "mtime": r"^\s*mtime\s*=\s*(.+)",
    "history_timestamp": r"^\s*history_timestamp\s*=\s*(\d+)",
    "comment": r"^\s*comment\s*=\s*(.+)",
    "exec_host": r"^\s*exec_host\s*=\s*(.+)",
    "exec_vnode": r"^\s*exec_vnode\s*=\s*(.+)",
    "ncpus_req": r"^\s*Resource_List\.ncpus\s*=\s*(\d+)",
    "ncpus_used": r"^\s*resources_used\.ncpus\s*=\s*(\d+)",
    "ngpus_req": r"^\s*Resource_List\.ngpus\s*=\s*(\d+)",
    "walltime_req": r"^\s*Resource_List\.walltime\s*=\s*(\S+)",
    "walltime_used": r"^\s*resources_used\.walltime\s*=\s*(\S+)",
    "cpupercent_used": r"^\s*resources_used\.cpupercent\s*=\s*(\d+)",
    "mem_used": r"^\s*resources_used\.mem\s*=\s*(\S+)",
}

COMPILED_PATTERNS = {
    key: re.compile(pattern, re.MULTILINE) for key, pattern in FIELD_PATTERNS.items()
}
NODE_PATTERN = re.compile(r"r\d{2}[cg]n\d{2}")


def extract_field(block, field_name):
    match = COMPILED_PATTERNS[field_name].search(block)
    return match.group(1).strip() if match else None


def parse_datetime(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, TIME_FORMAT)
    except ValueError:
        return None


def format_duration_seconds(total_seconds):
    if total_seconds is None:
        return None
    total_seconds = int(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def derive_queue_wait(job):
    qtime = parse_datetime(job.get("qtime"))
    stime = parse_datetime(job.get("stime"))
    if not (qtime and stime):
        return None, None
    wait_seconds = int((stime - qtime).total_seconds())
    if wait_seconds < 0:
        return None, None
    return wait_seconds, format_duration_seconds(wait_seconds)


def normalize_username(job):
    owner = job.get("job_owner")
    if owner:
        return owner.split("@", 1)[0]
    if job.get("euser"):
        return job["euser"]
    return None


def extract_nodes(job):
    hosts = []
    for field in ("exec_host", "exec_vnode"):
        value = job.get(field) or ""
        hosts.extend(NODE_PATTERN.findall(value))
    return sorted(set(hosts))


def derive_start_time(job):
    for field in START_TIME_CANDIDATES:
        value = job.get(field)
        if parse_datetime(value):
            return value, field
    return None, None


def parse_job_block(block, source_file):
    job = {field: extract_field(block, field) for field in COMPILED_PATTERNS}
    job["source_file"] = source_file
    job["user"] = normalize_username(job)
    job["nodes"] = extract_nodes(job)

    start_time, start_source = derive_start_time(job)
    queue_wait_seconds, queue_wait_time = derive_queue_wait(job)
    job["start_time"] = start_time
    job["start_time_source"] = start_source
    job["end_time"] = job.get("mtime") if parse_datetime(job.get("mtime")) else None
    job["queue_wait_seconds"] = queue_wait_seconds
    job["queue_wait_time"] = queue_wait_time
    job["has_queue_wait_time"] = queue_wait_seconds is not None
    job["has_required_times"] = bool(
        all(parse_datetime(job.get(field)) for field in REQUIRED_TIME_FIELDS)
        and start_time
    )

    return job


def stream_job_blocks(filepath):
    block_lines = []
    with open(filepath, "r", errors="ignore") as handle:
        for line in handle:
            if line.startswith("Job Id:") and block_lines:
                yield "".join(block_lines)
                block_lines = [line]
            else:
                block_lines.append(line)
    if block_lines:
        yield "".join(block_lines)


def merge_jobs(existing, incoming):
    for key, value in incoming.items():
        if key == "source_file":
            continue
        if not existing.get(key) and value is not None:
            existing[key] = value

    if existing.get("queue_wait_seconds") is None and incoming.get("queue_wait_seconds") is not None:
        existing["queue_wait_seconds"] = incoming["queue_wait_seconds"]
        existing["queue_wait_time"] = incoming.get("queue_wait_time")
        existing["has_queue_wait_time"] = incoming.get("has_queue_wait_time", False)

    sources = set(existing.get("source_files", []))
    sources.add(incoming["source_file"])
    existing["source_files"] = sorted(sources)

    existing_nodes = set(existing.get("nodes", []))
    existing_nodes.update(incoming.get("nodes", []))
    existing["nodes"] = sorted(existing_nodes)

    start_candidates = []
    for candidate in [
        (existing.get("start_time"), existing.get("start_time_source")),
        (incoming.get("start_time"), incoming.get("start_time_source")),
    ]:
        if parse_datetime(candidate[0]):
            start_candidates.append(candidate)
    if start_candidates:
        best_start = min(start_candidates, key=lambda item: parse_datetime(item[0]))
        existing["start_time"], existing["start_time_source"] = best_start

    end_candidates = [value for value in [existing.get("end_time"), incoming.get("end_time")] if parse_datetime(value)]
    if end_candidates:
        existing["end_time"] = max(end_candidates, key=parse_datetime)

    existing["has_required_times"] = bool(
        all(parse_datetime(existing.get(field)) for field in REQUIRED_TIME_FIELDS)
        and parse_datetime(existing.get("start_time"))
    )

    return existing


def extract_jobs_from_files(files, target_user):
    jobs_by_id = {}

    for filepath in files:
        for block in stream_job_blocks(filepath):
            job = parse_job_block(block, filepath)
            if job.get("user") != target_user:
                continue
            if not job.get("job_id"):
                continue

            job_id = job["job_id"]
            if job_id in jobs_by_id:
                jobs_by_id[job_id] = merge_jobs(jobs_by_id[job_id], job)
            else:
                job["source_files"] = [filepath]
                jobs_by_id[job_id] = job

    filtered_jobs = [job for job in jobs_by_id.values() if job.get("has_required_times")]
    filtered_jobs.sort(key=lambda item: parse_datetime(item["start_time"]) or datetime.max)
    return filtered_jobs, jobs_by_id


def build_summary(filtered_jobs, all_jobs):
    return {
        "target_user": TARGET_USER,
        "total_unique_jobs_found": len(all_jobs),
        "correlation_ready_jobs": len(filtered_jobs),
        "excluded_jobs_missing_required_times": len(all_jobs) - len(filtered_jobs),
        "jobs_with_queue_wait_time": sum(1 for job in filtered_jobs if job.get("has_queue_wait_time")),
        "required_time_fields": list(REQUIRED_TIME_FIELDS),
        "start_time_candidates": list(START_TIME_CANDIDATES),
        "queue_wait_rule": "Calculated from qtime and stime when both are available.",
        "date_range": {
            "first_start_time": filtered_jobs[0]["start_time"] if filtered_jobs else None,
            "last_end_time": filtered_jobs[-1]["end_time"] if filtered_jobs else None,
        },
    }


def main():
    filtered_jobs, all_jobs = extract_jobs_from_files(SOURCE_FILES, TARGET_USER)
    output = {
        "summary": build_summary(filtered_jobs, all_jobs),
        "jobs": filtered_jobs,
    }

    Path(OUTPUT_FILE).write_text(json.dumps(output, indent=4))

    print(f"Unique {TARGET_USER} jobs found: {len(all_jobs)}")
    print(f"Correlation-ready jobs written: {len(filtered_jobs)}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
