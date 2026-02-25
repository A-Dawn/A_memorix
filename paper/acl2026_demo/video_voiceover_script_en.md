# A_Memorix Demo Voiceover Script (Friendly, 2m30s)

## Recording Goal

Use this script as teleprompter text.  
Target duration: **2:20 to 2:30**.

## Timeline Script

### 00:00 - 00:12 (Title)

Hello everyone, and thanks for watching.  
I am Chen Xi, an independent researcher from China.  
In this demo, I will introduce A_Memorix, an API-first memory service for long-horizon NLP applications.

### 00:12 - 00:30 (Problem)

Many LLM applications can retrieve text, but they still struggle with three practical needs:  
time-aware retrieval, memory governance, and safe update-and-recovery operations.  
A_Memorix brings these capabilities into one standalone service.

### 00:30 - 00:50 (Architecture)

The system combines three stores:  
a FAISS-based vector store, a relation graph store, and a SQLite metadata store.  
On top of this architecture, we provide dual-path retrieval and strict minute-level temporal filtering.

### 00:50 - 01:12 (Service Health)

Now let us run a short live workflow.  
First, I start the service and verify health and readiness.

### 01:12 - 01:35 (Import Task)

Next, I create an asynchronous import task and poll its status.  
This is how new memory enters the system without blocking user requests.

### 01:35 - 01:58 (Temporal Query)

Then I run a temporal query through `/v1/query/time`,  
using fields such as `query`, `time_from`, `time_to`, and `top_k`.  
This gives semantically relevant results constrained by time.

### 01:58 - 02:15 (Memory Operation)

After that, I apply memory operations such as protect and reinforce,  
and then query again to show that memory lifecycle controls are part of the same API surface.

### 02:15 - 02:27 (Evidence)

For reliability, our in-tree tests currently pass 12 out of 12.  
In benchmark results, metadata temporal query reaches 826.8 QPS at 5k paragraphs with 1.515 ms p95.  
HTTP end-to-end temporal query reaches 121.2 QPS with 26.166 ms p95.

### 02:27 - 02:30 (Closing)

Thank you for watching.  
A_Memorix is available as a reproducible release package and demo artifact.

## On-Screen Checklist

1. Title card with project name and author.
2. One architecture figure.
3. Terminal: service startup + `/healthz` + `/readyz`.
4. Terminal: `/v1/import/tasks` create + status polling.
5. Terminal: `/v1/query/time` request and response.
6. Terminal: one `/v1/memory/*` operation.
7. Benchmark and test result snapshot.
8. Final slide with release link and video link.
