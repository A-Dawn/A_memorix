# A_Memorix: An API-First Hybrid Memory Service for Temporal and Graph-Aware Retrieval (ACL 2026 Demo Draft)

**Author**: ChenXI (CN)  
**Submission mode**: Personal / independent submission  
**Code repository**: https://github.com/A-Dawn

## Abstract

We present **A_Memorix**, an API-first memory service that unifies vector retrieval, graph retrieval, temporal filtering, and online memory maintenance in a single standalone system. The system is designed for long-horizon conversational and agentic applications where users need to ingest new knowledge continuously, retrieve it under semantic and temporal constraints, and revise memory states over time. A_Memorix combines (1) dual-path retrieval over dense vectors and knowledge graph relations, (2) sparse fallback retrieval and rank fusion for robustness, (3) minute-level temporal querying with explicit time-range semantics, and (4) lifecycle-oriented memory operations such as protect, reinforce, freeze, prune, and restore. The demo exposes both a new `/v1/*` API and a backward-compatible `/api/*` interface, plus a web visualization front-end. We demonstrate end-to-end usage covering asynchronous import, temporal search, memory interventions, person-profile querying, and transcript summarization. In local measurements, the metadata temporal path reaches 826.8 QPS at 5k paragraphs (p95 1.515 ms), while HTTP end-to-end `/v1/query/time` reaches 121.2 QPS at 5k paragraphs (p95 26.166 ms).  

## 1 Introduction

Many LLM-centric applications require a memory layer that is not only searchable, but also maintainable: old facts should decay, important facts should be reinforced, and deleted or frozen knowledge should remain auditable and recoverable. Existing pipelines often split these functions across disconnected tools, increasing integration cost and operational complexity.

This demo introduces **A_Memorix**, a standalone service focused on practical memory operations for NLP systems. The current release is API-specialized and runs as a single process:

```bash
python -m amemorix serve --config ./config.toml
```

The service includes:

- a modern `/v1/*` API surface for ingestion, retrieval, deletion, memory operations, person profile, and summary tasks;
- a backward-compatible `/api/*` surface for existing integrations and web UI behavior;
- OpenAI-compatible embedding/LLM endpoint support with provider-agnostic configuration.

## 2 System Overview

### 2.1 Runtime Architecture

A_Memorix is implemented as a FastAPI application with a shared runtime context:

- `amemorix/app.py`: app factory, middleware, router registration, startup/shutdown.
- `amemorix/bootstrap.py`: component construction for storage, retriever, threshold filter, profile service.
- `amemorix/task_manager.py`: asynchronous workers and periodic maintenance loops.

Data is persisted under three stores:

- **Vector store** (`data/vectors`): FAISS-backed append-only storage with SQ8 quantization and fallback flat index.
- **Graph store** (`data/graph`): sparse adjacency matrix with relation-hash mapping.
- **Metadata store** (`data/metadata`): SQLite metadata, temporal fields, FTS indexes, task tables, and recovery tables.

### 2.2 API Surfaces

The system provides two equivalent operation layers:

- `/v1/*` for new integrations (22 routes in current codebase).
- `/api/*` for compatibility with historical contracts (22 routes in current codebase).

This dual surface allows migration without forcing immediate client rewrites.

## 3 Core Technical Components

### 3.1 Ingestion and Storage

`ImportService` supports multiple modes (`text`, `paragraph`, `relation`, `json`, `file`) and writes:

1. paragraph records with optional temporal metadata;
2. relation/entity metadata and graph edges;
3. embeddings for both paragraph and relation content.

The vector store normalizes vectors and supports a transition from fallback flat indexing to SQ8-based indexing once sufficient training data is available. The metadata store includes schema migration logic for temporal fields, memory lifecycle fields, and task/person-profile tables.

### 3.2 Dual-Path Retrieval with Robust Fallback

`DualPathRetriever` executes retrieval across two complementary paths:

- **Paragraph path**: semantic vector retrieval over stored content.
- **Relation path**: entity-pivoted relation retrieval and relation-aware evidence expansion.

When dense retrieval is weak or unavailable, the system can invoke sparse retrieval (`SparseBM25Index`) via SQLite FTS5 and optional n-gram fallback. Candidate lists are merged using weighted RRF fusion. An optional Personalized PageRank reranker adjusts scores using entity-aware graph saliency.

### 3.3 Temporal Retrieval Semantics

For temporal search, the system enforces structured query formats:

- `YYYY/MM/DD`
- `YYYY/MM/DD HH:mm`

Temporal filtering uses interval-overlap semantics and computes effective time from `event_time_start`, `event_time_end`, or `event_time`; optionally, `created_at` can be used as fallback. This design supports both precise event windows and partially timestamped data.

### 3.4 Memory Lifecycle Operations

The maintenance loop performs periodic decay and state transitions:

- **decay** on edge weights;
- **freeze** for low-weight, unprotected relations;
- **prune** expired inactive relations into a recycle-bin table;
- **restore** from recycle bin back to active graph relation.

Manual endpoints (`/v1/memory/*`) expose `status`, `protect`, `reinforce`, and `restore`, enabling human-in-the-loop memory governance.

### 3.5 Async Tasks, Profiles, and Summarization

`TaskManager` manages async import and summary queues. Summary jobs transform transcript messages into structured summary/entity/relation payloads and ingest them as knowledge. Person profile functionality combines registry aliases, graph evidence, and retrieval evidence to build snapshot-based profiles with optional manual overrides.

## 4 Demo Scenario

Our demo session follows five stages:

1. **Start service and verify readiness** (`/healthz`, `/readyz`).
2. **Submit import task** (`POST /v1/import/tasks`) and poll status.
3. **Issue temporal query** (`POST /v1/query/time`) with optional person/source constraints.
4. **Apply memory intervention** (`/v1/memory/protect` or `/v1/memory/reinforce`) and observe state changes.
5. **Run summary/profile tasks** (`/v1/summary/tasks`, `/v1/person/query`) and inspect surfaced evidence.

A local regression run in the repository logs shows successful startup, import task execution, and temporal query responses (HTTP 200), consistent with this scenario.

## 5 Micro-Benchmark and In-Tree Tests

### 5.1 Temporal Query Micro-Benchmark

We benchmarked `MetadataStore.query_paragraphs_temporal` using
`benchmarks/benchmark_temporal_query.py` on a local Windows 11 machine
(Intel i7-13650HX, 31.69 GB RAM, Python 3.12.7). Results are summarized below.

| Case | paragraphs | queries | person filter ratio | qps | mean (ms) | p95 (ms) | p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small_seed7 | 1200 | 120 | 0.30 | 3046.783 | 0.319 | 0.454 | 0.560 |
| medium_seed42 | 5000 | 300 | 0.25 | 826.801 | 1.196 | 1.515 | 1.670 |
| medium_noperson_seed42 | 5000 | 300 | 0.00 | 1081.195 | 0.917 | 1.275 | 1.328 |
| large_seed42 | 10000 | 500 | 0.25 | 425.640 | 2.332 | 2.976 | 3.225 |

These runs indicate low-millisecond temporal lookup latency at current scales, with expected slowdown under larger stores and person-filtered queries.

### 5.2 `/v1/query/time` End-to-End Benchmark

We additionally measured the HTTP end-to-end path using
`benchmarks/benchmark_v1_time_e2e.py`, which synthesizes temporal data,
starts a local service instance, and benchmarks `POST /v1/query/time`.

| Case | paragraphs | queries | person filter ratio | qps | mean (ms) | p95 (ms) | p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e_small_seed7 | 800 | 80 | 0.30 | 156.131 | 6.358 | 24.160 | 27.345 |
| e2e_medium_seed42 | 3000 | 200 | 0.25 | 132.924 | 7.502 | 25.086 | 27.529 |
| e2e_medium_noperson_seed42 | 3000 | 200 | 0.00 | 138.895 | 7.189 | 23.149 | 28.898 |
| e2e_large_seed42 | 5000 | 300 | 0.25 | 121.208 | 8.247 | 26.166 | 29.620 |

Compared with storage-path micro-benchmarks, E2E numbers reflect realistic framework, serialization, and API-layer overhead.

### 5.3 In-Tree Test Status

The repository now includes in-tree tests under `tests/`, with current run status:

- `python -m pytest -q` -> `12 passed`

The tests cover time parsing, temporal filtering behavior, summary import flow, and unified search execution service behavior.

## 6 Implementation Status and Reliability Signals

Current repository signals include:

- release `0.6.1` with API-specialized architecture and contract synchronization;
- changelog-reported full regression pass (`60 passed / 0 failed`) from prior release verification;
- runnable CLI and FastAPI docs endpoints (`/docs`, `/openapi.json`);
- configurable auth middleware with write-protected operations by default.
- Docker smoke validation on 2026-02-24: container startup OK, `healthz/readyz` OK, unauthorized write blocked (`401`), authorized write and `/v1/query/time` available.

These signals indicate deployment readiness for demo purposes, while full benchmark-scale evaluation remains future work.

## 7 Limitations and Ongoing Work

- Benchmark workloads are synthetic; real multi-tenant traffic and heterogeneous data distributions may shift the latency profile.
- Demo reproducibility for full retrieval quality still depends on external OpenAI-compatible providers.

## 8 Ethics Statement (Draft)

A_Memorix stores user-provided textual memory and derived graph relations; this can include personal or sensitive information. We recommend:

- strict token-based access control in production;
- data minimization and retention policies for transcript content;
- user-visible deletion and restoration controls with auditability;
- explicit consent before enabling profile aggregation.

The system provides mechanisms for deletion, soft deletion, and restoration, but policy compliance still depends on deployment settings and operator governance.

## 9 Reproducibility Notes

The demo can be reproduced with:

1. dependency install (`pip install -r requirements.txt`);
2. config preparation from `config.toml.example`;
3. service start (`python -m amemorix serve --config ./config.toml`);
4. test run (`python -m pytest -q`);
5. metadata micro-benchmark (`python benchmarks/benchmark_temporal_query.py --paragraphs 5000 --queries 300`);
6. HTTP E2E benchmark (`python benchmarks/benchmark_v1_time_e2e.py --paragraphs 3000 --queries 200`);
7. API invocation via `/v1/*` endpoints documented in `DEVELOPER_API_GUIDE.md`.
8. Docker smoke test (build + run + auth/readiness checks), see `benchmark.md` and `benchmarks/results/docker_smoke_20260224.json`.

## 10 Camera-Ready TODO Checklist

- [ ] Replace title and abstract wording after author feedback.
- [x] Add author block and submission mode metadata (`ChenXI (CN)`, personal submission).
- [x] Add public code URL (`https://github.com/A-Dawn`).
- [ ] Add demo video URL in abstract and submission form.
- [ ] Add screenshot figure(s) for web UI and task status flow.
- [ ] Convert to ACL LaTeX format and finalize references.
