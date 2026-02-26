# A_Memorix Demo Voiceover Script
### 00:00 - 00:12 (Title)

Hello, I am Chen Xi, an independent researcher from China.  
This demo presents A_Memorix, an API-first memory service for long-horizon NLP applications.

### 00:12 - 00:30 (Problem)

Many LLM applications retrieve text, but still lack practical memory control.  
We focus on three gaps: time-aware retrieval, lifecycle governance, and safe update-and-recovery.  
Our goal is a standalone runtime that is reproducible and easy to integrate.

### 00:30 - 00:50 (Architecture)

The system combines three persistent stores:  
a FAISS vector store, a relation graph store, and SQLite metadata.  
It supports dual-path retrieval, minute-level temporal filtering, `/v1` for new integration,  
and `/api` for backward compatibility.

### 00:50 - 01:02 (Demo Reset + Service Health)

Now I start the live workflow.  
First, I optionally clean previous demo records by source.  
Then I verify `/healthz` and `/readyz`.

### 01:02 - 01:20 (Import Task: Paragraph)

Next, I create an asynchronous paragraph import task with `/v1/import/tasks`,  
and poll task status until it succeeds.  
This stage shows non-blocking ingestion with stable task management.

### 01:20 - 01:36 (Import Task: Relation)

Then I create a relation import task and wait for completion.  
This provides an explicit relation target for later lifecycle operations.

### 01:36 - 01:52 (Temporal Query Before Action)

Then I run `/v1/query/time` using `query`, `time_from`, `time_to`, and `source`.  
The response returns semantically relevant memories constrained by minute-level time boundaries.

### 01:52 - 02:12 (Memory Status + Protect)

After that, I call `/v1/memory/status`, then apply `/v1/memory/protect`,  
and check status again.  
The `ttl_protected_relations` value increases, showing active lifecycle control in the same API surface.

### 02:12 - 02:20 (Temporal Query After Action)

Finally, I run the temporal query again.  
This closes the loop of import, retrieval, lifecycle control, and retrieval validation.

### 02:20 - 02:27 (Evidence)

For evidence, all 12 in-tree tests pass,  
and benchmarks report reproducible throughput and latency.

### 02:27 - 02:30 (Closing)

Thank you.  
Artifacts are available in the project release package.
