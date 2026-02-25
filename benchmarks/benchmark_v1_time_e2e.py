from __future__ import annotations

import argparse
import json
import random
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.storage.metadata_store import MetadataStore

NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def request_json(
    *,
    method: str,
    url: str,
    body: Optional[dict[str, Any]] = None,
    token: Optional[str] = None,
    timeout: float = 10.0,
) -> tuple[int, dict[str, Any]]:
    data: Optional[bytes] = None
    headers: dict[str, str] = {}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url=url, data=data, headers=headers, method=method.upper())
    try:
        with NO_PROXY_OPENER.open(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="ignore").strip()
            payload = json.loads(text) if text else {}
            return int(resp.status), payload
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="ignore").strip()
        payload = {}
        if text:
            try:
                payload = json.loads(text)
            except Exception:
                payload = {"raw": text}
        return int(exc.code), payload


def write_temp_config(
    *,
    config_path: Path,
    data_dir: Path,
    host: str,
    port: int,
    token: str,
) -> None:
    config_text = f"""
[server]
host = "{host}"
port = {port}
workers = 1

[auth]
enabled = true
write_tokens = ["{token}"]
read_tokens = []
protect_read_endpoints = false

[storage]
data_dir = "{data_dir.as_posix()}"

[embedding]
dimension = 1024
auto_detect_dimension = false
batch_size = 32
max_concurrent = 2
quantization_type = "int8"

[embedding.openapi]
base_url = "http://127.0.0.1:9/v1"
api_key = "bench-key"
model = "text-embedding-3-large"
timeout_seconds = 1
max_retries = 1

[tasks]
import_workers = 1
summary_workers = 1
queue_maxsize = 128

[cors]
allow_origins = []
""".strip()
    config_path.write_text(config_text + "\n", encoding="utf-8")


def synthesize_temporal_data(
    *,
    data_dir: Path,
    paragraphs: int,
    seed: int,
    person_frequency: int = 4,
) -> dict[str, Any]:
    rng = random.Random(seed)
    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    store = MetadataStore(data_dir=metadata_dir)
    store.connect()
    base_dt = datetime(2026, 1, 1, 0, 0)
    person_pool = [f"user_{idx:03d}" for idx in range(80)]
    created = 0
    started = time.perf_counter()
    try:
        for idx in range(paragraphs):
            offset_minutes = rng.randint(0, 60 * 24 * 45)
            event_dt = base_dt + timedelta(minutes=offset_minutes)
            para_hash = store.add_paragraph(
                content=f"e2e benchmark paragraph {idx}",
                source=f"bench:e2e:{idx % 12}",
                knowledge_type="factual",
                time_meta={"event_time": event_dt.strftime("%Y/%m/%d %H:%M")},
            )
            if person_frequency > 0 and idx % person_frequency == 0:
                person_name = person_pool[idx % len(person_pool)]
                store.add_entity(name=person_name, source_paragraph=para_hash)
            created += 1
    finally:
        store.close()
    ingest_ms = (time.perf_counter() - started) * 1000.0
    return {
        "paragraphs": created,
        "ingest_total_ms": round(ingest_ms, 3),
        "ingest_paragraphs_per_sec": round(created / max(1e-9, ingest_ms / 1000.0), 3),
        "person_pool_size": len(person_pool),
    }


def run_e2e_benchmark(
    *,
    paragraphs: int,
    queries: int,
    query_window_minutes: int,
    query_limit: int,
    person_filter_ratio: float,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    base_dt = datetime(2026, 1, 1, 0, 0)
    max_offset = max(1, 60 * 24 * 45 - query_window_minutes - 1)
    person_pool = [f"user_{idx:03d}" for idx in range(80)]

    with tempfile.TemporaryDirectory(prefix="amemorix_e2e_") as tmp:
        tmp_dir = Path(tmp)
        data_dir = tmp_dir / "data"
        config_path = tmp_dir / "config.toml"
        logs_dir = tmp_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = logs_dir / "server.stdout.log"
        stderr_path = logs_dir / "server.stderr.log"

        ingest = synthesize_temporal_data(
            data_dir=data_dir,
            paragraphs=paragraphs,
            seed=seed,
        )

        host = "127.0.0.1"
        port = find_free_port()
        token = "bench-token"
        base_url = f"http://{host}:{port}"
        write_temp_config(config_path=config_path, data_dir=data_dir, host=host, port=port, token=token)

        with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("w", encoding="utf-8") as err_f:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "amemorix",
                    "serve",
                    "--config",
                    str(config_path),
                    "--host",
                    host,
                    "--port",
                    str(port),
                ],
                cwd=str(ROOT),
                stdout=out_f,
                stderr=err_f,
                text=True,
            )

            try:
                deadline = time.time() + 60.0
                while time.time() < deadline:
                    if proc.poll() is not None:
                        stdout_tail = stdout_path.read_text(encoding="utf-8", errors="ignore")[-3000:]
                        stderr_tail = stderr_path.read_text(encoding="utf-8", errors="ignore")[-3000:]
                        raise RuntimeError(
                            "server process exited before healthz ready.\n"
                            f"stdout_tail:\n{stdout_tail}\n\nstderr_tail:\n{stderr_tail}"
                        )
                    try:
                        status, payload = request_json(
                            method="GET",
                            url=f"{base_url}/healthz",
                            body=None,
                            token=None,
                            timeout=3.0,
                        )
                        if status == 200 and payload.get("status") == "ok":
                            break
                    except Exception:
                        pass
                    time.sleep(0.25)
                else:
                    raise TimeoutError("healthz timeout")

                warmup = min(30, max(5, queries // 10))
                for _ in range(warmup):
                    start_offset = rng.randint(0, max_offset)
                    start_dt = base_dt + timedelta(minutes=start_offset)
                    end_dt = start_dt + timedelta(minutes=query_window_minutes)
                    body: dict[str, Any] = {
                        "query": "",
                        "time_from": start_dt.strftime("%Y/%m/%d %H:%M"),
                        "time_to": end_dt.strftime("%Y/%m/%d %H:%M"),
                        "top_k": query_limit,
                    }
                    if rng.random() < person_filter_ratio:
                        body["person"] = person_pool[rng.randint(0, len(person_pool) - 1)]
                    status, _ = request_json(
                        method="POST",
                        url=f"{base_url}/v1/query/time",
                        body=body,
                        token=token,
                        timeout=8.0,
                    )
                    if status != 200:
                        raise RuntimeError(f"warmup request failed: status={status}")

                latencies_ms: list[float] = []
                total_hits = 0
                measured_start = time.perf_counter()
                for _ in range(queries):
                    start_offset = rng.randint(0, max_offset)
                    start_dt = base_dt + timedelta(minutes=start_offset)
                    end_dt = start_dt + timedelta(minutes=query_window_minutes)

                    body = {
                        "query": "",
                        "time_from": start_dt.strftime("%Y/%m/%d %H:%M"),
                        "time_to": end_dt.strftime("%Y/%m/%d %H:%M"),
                        "top_k": query_limit,
                    }
                    if rng.random() < person_filter_ratio:
                        body["person"] = person_pool[rng.randint(0, len(person_pool) - 1)]

                    t0 = time.perf_counter()
                    status, payload = request_json(
                        method="POST",
                        url=f"{base_url}/v1/query/time",
                        body=body,
                        token=token,
                        timeout=8.0,
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    if status != 200:
                        raise RuntimeError(f"query failed: status={status}, payload={payload}")
                    total_hits += int(payload.get("count", 0) or 0)
                    latencies_ms.append(elapsed_ms)

                measured_elapsed_s = max(1e-9, time.perf_counter() - measured_start)
                qps = queries / measured_elapsed_s

                return {
                    "seed": seed,
                    "paragraphs": paragraphs,
                    "queries": queries,
                    "query_window_minutes": query_window_minutes,
                    "query_limit": query_limit,
                    "person_filter_ratio": round(person_filter_ratio, 3),
                    "ingest": {
                        "total_ms": ingest["ingest_total_ms"],
                        "paragraphs_per_sec": ingest["ingest_paragraphs_per_sec"],
                    },
                    "query": {
                        "total_sec": round(measured_elapsed_s, 6),
                        "qps": round(qps, 3),
                        "mean_ms": round(mean(latencies_ms), 3),
                        "p50_ms": round(percentile(latencies_ms, 50), 3),
                        "p95_ms": round(percentile(latencies_ms, 95), 3),
                        "p99_ms": round(percentile(latencies_ms, 99), 3),
                        "avg_hits": round(total_hits / max(1, queries), 3),
                    },
                }
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP end-to-end benchmark for /v1/query/time.")
    parser.add_argument("--paragraphs", type=int, default=5000, help="number of synthetic paragraphs")
    parser.add_argument("--queries", type=int, default=300, help="number of measured HTTP requests")
    parser.add_argument("--query-window-minutes", type=int, default=120, help="temporal window in minutes")
    parser.add_argument("--query-limit", type=int, default=50, help="top_k for each /v1/query/time call")
    parser.add_argument(
        "--person-filter-ratio",
        type=float,
        default=0.25,
        help="fraction of requests that include person filter (0.0 - 1.0)",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_e2e_benchmark(
        paragraphs=max(100, int(args.paragraphs)),
        queries=max(10, int(args.queries)),
        query_window_minutes=max(1, int(args.query_window_minutes)),
        query_limit=max(1, int(args.query_limit)),
        person_filter_ratio=max(0.0, min(1.0, float(args.person_filter_ratio))),
        seed=int(args.seed),
    )

    print("=== A_Memorix /v1/query/time E2E Benchmark ===")
    print(f"Run at: {datetime.now().isoformat(timespec='seconds')}")
    print(
        "Ingest: "
        f"{result['ingest']['paragraphs_per_sec']} paragraphs/s, "
        f"total={result['ingest']['total_ms']} ms"
    )
    print(
        "Query: "
        f"qps={result['query']['qps']}, "
        f"mean={result['query']['mean_ms']} ms, "
        f"p50={result['query']['p50_ms']} ms, "
        f"p95={result['query']['p95_ms']} ms, "
        f"p99={result['query']['p99_ms']} ms, "
        f"avg_hits={result['query']['avg_hits']}"
    )
    print()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
