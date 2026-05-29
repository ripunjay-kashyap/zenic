# Ship Zenic to Hugging Face Spaces — Deployed Demo

**Date:** 2026-05-29
**Workstream:** A — Ship it (deploy foundation)
**Status:** Approved, pending implementation plan

## Goal

Produce a reachable, working Hugging Face Spaces URL running Zenic with all 6
intents against Qdrant Cloud. The deliverable is a **deployed demo** we can look
at together — nothing more.

## Out of scope (deferred to the next cycle)

After we review the live demo, separate specs will cover production-grade
hardening and feature work. Captured backlog (each item to be scoped into its
own spec where it warrants one):

**Reliability & performance**
- Latency optimization
- Groq rate-limit handling (100k TPD free tier) + retries
- Caching
- Observability / structured logging / cost+latency metrics

**Engineering hygiene**
- CI/CD
- Repo cleanup (root clutter: `test-*.png`, `debug_*.py`, `tmp_*.txt`)

**Features**
- Profile creation (collect/persist user profile fields)
- PDF generation with proper per-intent templates: workout splits,
  nutritional / supplement splits, weekly summaries
- Frontend switch from Streamlit to React (**large — own spec**; revisits the
  current Streamlit + LangGraph + Qdrant architecture, likely splitting the app
  into a React frontend + an API backend exposing the agent graph)

## Deploy mechanism (corrected from roadmap)

The CLAUDE.md roadmap and memory describe deploying via
`podman push registry.hf.co/...`. **This path does not exist.** Per the official
HF docs, a Docker Space is deployed by committing a `Dockerfile` + a `README.md`
with YAML metadata to the Space's git repo; **HF builds the image on their side**.
`registry.hf.space` is where HF lets you *pull* the image they built — it is an
output, not a deploy input.

Corrected shape:

- **Local Podman build + `podman run`** → smoke-test the container locally before
  pushing (fast feedback, catches issues before HF's slower build).
- **Deploy = `git push` to the HF Space repo** → HF rebuilds from the same
  `Dockerfile` and runs it.

Sources:
- https://huggingface.co/docs/hub/spaces-sdks-docker
- https://huggingface.co/docs/hub/spaces-run-with-docker

### HF Spaces hard constraints

- App must listen on **port 7860** (or declare `app_port` in README YAML).
- Container runs as **UID 1000** — Dockerfile must `useradd -m -u 1000 user`,
  use `COPY --chown=user`, and set `HF_HOME` so the pre-baked model cache is
  readable by that user.
- **Secrets** (`GROQ_API_KEY`, `USDA_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`,
  `GOOGLE_API_KEY`, `ENV=production`) are set in **Space Settings**, never
  committed.
- The `/data` runtime volume is **not** available at build time — fine for us,
  since `bm25_corpus.json` is `COPY`'d from the repo at build time.

## Grounding facts (verified against the code)

- `data/bm25_corpus.json` (9.4 MB) is already un-ignored in `.gitignore`
  (`!data/bm25_corpus.json`). It is a list of **10,201 chunks**, each with
  `id`, `text`, `metadata`. This is the source of truth for rebuilding Qdrant.
- Embedding model `BAAI/bge-small-en-v1.5` → **384-dim** vectors, cosine distance.
- Reranker `BAAI/bge-reranker-base`.
- Models load lazily from HF Hub on first request (`pipeline.py:45,53`) — must be
  pre-baked at build time to avoid cold-start downloads.
- `_QdrantAdapter.__init__` (`vector_store.py:88`) **does not create** the
  collection. The migration script must create it.
- `_QdrantAdapter.upsert` (`vector_store.py:91`) passes the raw string `id` to
  `PointStruct(id=...)`. Qdrant requires unsigned int or UUID — this would fail
  on string IDs and must be fixed.
- Metadata in `bm25_corpus.json` may be a **stringified dict**
  (`"{'source': 'USDA', ...}"`); the migration must parse it before upserting.

## Deliverables

1. **`scripts/migrate_to_qdrant.py`** (new) — one-time, run locally:
   - Read `data/bm25_corpus.json` (utf-8).
   - Load `SentenceTransformer(EMBED_MODEL)`; embed `text` in batches.
   - Create collection `zenic_knowledge` if missing: **size 384, distance Cosine**.
   - Map string `id` → `uuid5`; store original `id` + parsed `metadata` + `text`
     in the payload.
   - Idempotent (safe to re-run).

2. **Fix `_QdrantAdapter.upsert`** (`vector_store.py:91`) — convert string IDs →
   `uuid5` to match the migration and avoid failures. Search reads payload, never
   point-ID, so this change is safe.

3. **`Dockerfile`** (new):
   - `FROM python:3.12-slim`.
   - `RUN useradd -m -u 1000 user`; `USER user`; `WORKDIR $HOME/app`;
     `ENV HF_HOME=$HOME/.cache/huggingface`.
   - Install **CPU-only torch** + `requirements.txt` (keeps image lean).
   - Pre-bake `bge-small-en-v1.5` + `bge-reranker-base` via
     `RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"`.
   - `COPY --chown=user` app code + `data/bm25_corpus.json`.
   - `ENV ENV=production`.
   - Run Streamlit on **7860**:
     `streamlit run zenic/ui/app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true`.

4. **`.dockerignore`** (new) — exclude `chroma_db/`, `test-*.png`, `tests/`,
   `scripts/` (except those needed at runtime — none are), `.git`, `eval_*`,
   `OVERVIEW/`, `TESTING/`, `*.png`, etc., so the build context is small.

5. **Space `README.md` YAML front-matter** — `sdk: docker`, `app_port: 7860`,
   `title`, `emoji`, `colorFrom`, `colorTo`.

6. **`.env.example` fix** — replace the stale "Streamlit Community Cloud"
   comment (`.env.example:17`) with a Hugging Face Spaces reference.

## Runbook (the plan will expand this)

1. Provision a Qdrant Cloud cluster (free tier) → obtain `QDRANT_URL` +
   `QDRANT_API_KEY`; put them in local `.env`.
2. Run `PYTHONPATH=. python scripts/migrate_to_qdrant.py` locally (one-time,
   ~10,201 embeddings).
3. Write `Dockerfile` + `.dockerignore`; `podman build` then
   `podman run --env-file .env -p 7860:7860` → **local smoke test** all 6 intents.
4. Create the HF Space (Docker SDK); set Space secrets; `git push` → HF builds.
5. **Smoke test the live URL** across all 6 intents — this is the deployed demo
   we review before planning the production-grade cycle.

## Success criteria

- Live HF Spaces URL loads the Zenic UI.
- All 6 intents return correct, cited answers against Qdrant Cloud (not local
  ChromaDB).
- No cold-start model download on first request (models pre-baked).
