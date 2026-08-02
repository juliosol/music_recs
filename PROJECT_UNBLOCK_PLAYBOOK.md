# Music Recommender Unblock Playbook

## Goal
Turn this project from a small-demo dataset into a production-viable recommendation service with strong coverage, low latency, and reliable updates.

---

## 1) Decide the product target first (Week 0)

Before collecting more data, lock the scope:

- Target audience: global users
- Primary use case: song discovery and mood recommendations
- Freshness target: daily
- Latency target: p95 response time < 5s
- Catalog target: 1M tracks (initial production target)

Week 0 status: completed. Product scope is now fixed to the targets above.

Why: billions of entries is expensive and unnecessary at this stage. You can get excellent quality with smart retrieval and embedding indexes at much smaller scale.

---

## 2) Stabilize the current app path (Week 1)

## 2.1 Keep request-time inference fast
Do not perform heavy audio download/feature extraction inside a user request.

- Request path should only:
  - parse input playlist
  - fetch precomputed vectors from storage
  - run ANN similarity search
  - apply diversity/business filters

## 2.2 Add graceful fallback for unknown tracks
If a playlist track is missing from your catalog:

- Use metadata-only fallback (title/channel/tags/text embeddings)
- Queue missing tracks for background enrichment
- Return recommendations immediately from known tracks

## 2.3 Add observability
Log and track:

- request duration by stage
- percent of tracks found in catalog
- fallback rate
- recommendation click/play metrics

---

## 3) Grow catalog the right way (Weeks 1-4)

## 3.1 Multi-source ingestion strategy
Use multiple data sources instead of only one static crawler loop.

- Seed sources:
  - curated genre lists
  - regional charts
  - user-submitted playlists
  - long-tail exploration via query expansion

- Expansion sources:
  - related videos/channels from seeds
  - co-occurrence from user playlist overlap
  - trending APIs where available

## 3.2 Build an incremental data pipeline
Treat collection as recurring ETL, not one-time script runs.

Pipeline stages:

1. Ingest IDs + metadata
2. Deduplicate and canonicalize
3. Feature extraction (audio/text/popularity)
4. Embed to vector space
5. Index build/update
6. Quality checks
7. Publish snapshots

## 3.3 Use layered features
Do not rely only on raw audio.

- Audio features: tempo, energy, valence, danceability proxies
- Text features: title/channel/description embeddings
- Behavioral features: popularity trajectory, recency, skip-like heuristics
- Graph features: playlist co-occurrence and artist relations

---

## 4) Recommended architecture for scale (Month 2)

## 4.1 Storage split
- Raw lake: object store for raw dumps and intermediate files
- Feature store: columnar tables (Parquet/Delta/Iceberg)
- Online serving DB: compact table keyed by track_id
- Vector index: FAISS (single node) -> Milvus/Weaviate/Pinecone (distributed)

## 4.2 Retrieval design (2-stage)
1. Candidate generation: ANN search over embeddings (top 500-2k)
2. Reranking: blend similarity, diversity, freshness, popularity, novelty

Scoring example:

score = w1*sim + w2*diversity + w3*freshness + w4*novelty + w5*popularity

## 4.3 Freshness model
- Daily full refresh for core catalog
- Streaming/incremental updates for new tracks and trending shifts
- Blue/green index rollout for zero-downtime updates

---

## 5) Data quality and trust (always on)

Add automatic checks at publish time:

- duplicate rate
- broken URL/id rate
- missing essential feature rate
- distribution drift vs previous snapshot
- outlier detection for bogus metadata

Block publish when quality thresholds fail.

---

## 6) Legal/compliance guardrails (critical)

For a public app, make policy decisions early:

- Respect platform ToS for data usage and scraping/download behavior
- Store only allowed metadata/features if required
- Implement content takedown and removal workflows
- Keep audit logs for ingestion provenance

This is often a bigger blocker than pure engineering scale.

---

## 7) Concrete milestones for this repo

## Milestone A (1-2 weeks): Reliable MVP at 100k+
- Background enrichment job (no heavy work in request path)
- Metadata fallback enabled
- Request timeout safeguards
- Vector index for fast retrieval
- Dashboard for latency and coverage

Exit criteria:
- p95 latency < 2s
- >85% playlist-track match rate
- recommendation request success >99%

## Milestone B (3-6 weeks): Public beta at 1M+
- Incremental ingestion scheduler
- Better embeddings and reranker
- A/B testing harness for ranking weights
- Human quality evaluation set

Exit criteria:
- measurable lift in CTR/save rate
- stable daily ingestion and publish

## Milestone C (2-3 months): Production hardening
- distributed vector database
- canary index rollouts
- autoscaling workers
- robust abuse/rate limiting

Exit criteria:
- SLOs met for 30 days
- low operational toil

---

## 8) Should you build billions of entries now?

Short answer: no.

Better path:

- Start 100k -> 1M high-quality entries
- Improve retrieval + ranking quality
- Expand where demand proves value

In recommendation systems, quality of representation and ranking usually beats raw catalog size at early stages.

---

## 9) Immediate next commands for this project

From project root:

1. Validate baseline:
   - conda run -n music_recs python test_youtube_system.py

2. Run/extend dataset collection in batches:
   - conda run -n music_recs python collect_youtube_dataset.py --max-per-playlist 200 --include-popular --regions US,GB,CA

3. Add periodic runs (cron/CI) and append mode:
   - conda run -n music_recs python collect_youtube_dataset.py --append --include-popular

4. Launch app:
   - cd recommendation_app
   - conda run -n music_recs python start_youtube.py

---

## 10) Technical debt items to prioritize in code

- Move all long-running extraction out of request handlers
- Add explicit request timeouts and user-facing progress/error responses
- Cache playlist extraction results keyed by playlist_id + recency window
- Persist user feedback events for future model improvement
- Add offline eval scripts (precision@k, diversity, novelty, coverage)

---

## Summary

To unblock this project for real users:

- optimize for speed and reliability first,
- grow to 100k-1M quality tracks with incremental pipelines,
- use vector retrieval + reranking,
- enforce quality and compliance checks,
- and only then scale further if usage justifies it.

---

## 11) Recommended Next Moves (Short-Term Execution)

These are the immediate follow-up actions after Part 2.1 implementation.

### 11.1 Graceful fallback for partial/empty playlist extraction (Part 2.2)
- If some playlist tracks fail extraction, continue with successful tracks.
- If primary recommendation output is empty due to weak ID overlap:
   - use a feature-space fallback based on normalized vectors,
   - then fallback to popular tracks as a final safety net.
- Always return recommendations when the catalog is available.

### 11.2 Queue unknown tracks for offline enrichment
- Persist missing track IDs from user playlists to an enrichment queue file/table.
- Run a background job to enrich those IDs (metadata, features, embeddings) outside request time.
- Re-index enriched tracks in the next daily refresh.

### 11.3 Add timeout and stage-level guardrails
- Introduce per-stage timeout budgets (playlist fetch, detail fetch, ranking).
- Return a friendly message when a stage times out, while still serving fallback recommendations.
- Keep p95 latency under 5s for online requests.

### 11.4 Add minimal observability to measure quality and reliability
- Log: request duration, extraction success count, fallback mode used.
- Track: playlist-track match rate, recommendation success rate, error rate.
- Use these metrics to decide where to invest next (catalog expansion vs ranking quality).

---

## 12) Implementation Status (Current)

### 12.1 Completed in code
- Part 2.1 completed:
   - request-time fast path avoids heavy audio downloads in `/recommend`.
- Part 2.2 completed:
   - graceful fallback chain implemented (feature-space fallback, then popular fallback).
   - missing playlist IDs are queued to `data_extraction/missing_playlist_tracks.csv`.
- Part 2.3 completed:
   - observability metrics added to `/health`.
   - stage timing captured (extraction/ranking/formatting/total).
   - event ingestion endpoint added (`POST /events`) for clicks/copy/download.

### 12.2 3.1 groundwork completed
- Added source-aware ingestion in collector with provenance fields:
   - `source_type`, `source_ref`, `ingested_at`.
- Added config-driven seed source ingestion via:
   - `data_extraction/seed_sources.json`.
- Added expansion-source query capabilities:
   - related videos expansion from seed IDs,
   - channel-based expansion from seed channels.
- Collector supports new 3.1 options:
   - `--seed-source-config`
   - `--expand-related`
   - `--expand-from-channels`
   - `--related-per-seed`
   - `--videos-per-channel`
   - `--max-seed-expansion-base`

### 12.3 Immediate next execution commands (3.1)
1. Dry run seed sources only:
    - `conda run -n music_recs python collect_youtube_dataset.py --seed-source-config data_extraction/seed_sources.json --skip-curated --append`
2. Enable related-video expansion:
    - `conda run -n music_recs python collect_youtube_dataset.py --seed-source-config data_extraction/seed_sources.json --expand-related --related-per-seed 5 --max-seed-expansion-base 100 --append`
3. Enable channel expansion:
    - `conda run -n music_recs python collect_youtube_dataset.py --seed-source-config data_extraction/seed_sources.json --expand-from-channels --videos-per-channel 10 --max-seed-expansion-base 100 --append`

### 12.4 Future improvements to prioritize
- Add co-occurrence expansion from user playlist overlap graphs.
- Add source-level quality scoring and source pruning rules.
- Add incremental scheduler (daily) with per-source quotas and retry policy.
- Add canonical artist/title normalization keys for stronger deduplication.
- Add policy/compliance filters before publish snapshots.
