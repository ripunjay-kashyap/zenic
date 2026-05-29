# Ship Zenic to HF Spaces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Project rule — commits:** This project's owner runs all `git commit` commands themselves. The agent executing this plan must `git add` the listed files and then **surface the commit command for the user to run** — do NOT run `git commit` autonomously.

**Goal:** Deploy a working Zenic demo to Hugging Face Spaces (Docker SDK), serving all 6 intents against Qdrant Cloud.

**Architecture:** A one-time local migration script rebuilds the production Qdrant collection from `data/bm25_corpus.json` (the source of truth: 10,201 chunks of `id`/`text`/`metadata`). A `Dockerfile` pre-bakes the embedding + reranker models and the BM25 corpus so there is no cold-start download. Deployment is `git push` to the HF Space repo — HF builds the image; Podman is used only for a local smoke test before pushing.

**Tech Stack:** Python 3.12, sentence-transformers (bge-small-en-v1.5 @ 384-dim, bge-reranker-base), qdrant-client, Streamlit, Podman, Hugging Face Spaces (Docker SDK).

---

## File Structure

- `zenic/rag/vector_store.py` (modify) — add a deterministic `string id → uuid5` point-ID helper; fix `_QdrantAdapter.upsert` to use it and stash the original id in the payload; add `_QdrantAdapter.ensure_collection(vector_size)` so the collection is created before upsert (closes the existing gap where the adapter assumes it exists).
- `scripts/migrate_to_qdrant.py` (create) — one-time migration: read corpus → embed in batches → ensure collection → upsert.
- `tests/pillar1/test_vector_store.py` (create) — unit test for the pure point-ID helper.
- `Dockerfile` (create) — HF-Spaces-compliant image (UID 1000, port 7860, pre-baked models + corpus).
- `.dockerignore` (create) — keep build context small.
- `README.md` (modify) — prepend HF Space YAML front-matter.
- `.env.example` (modify) — fix stale "Streamlit Community Cloud" reference.

---

## Task 1: Point-ID helper + Qdrant adapter fixes

**Files:**
- Modify: `zenic/rag/vector_store.py`
- Test: `tests/pillar1/test_vector_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/pillar1/test_vector_store.py`:

```python
import uuid
from zenic.rag.vector_store import _to_point_id


def test_to_point_id_is_deterministic():
    assert _to_point_id("usda_167782") == _to_point_id("usda_167782")


def test_to_point_id_differs_per_input():
    assert _to_point_id("usda_167782") != _to_point_id("usda_167783")


def test_to_point_id_is_valid_uuid_string():
    out = _to_point_id("issn_protein_01")
    # must not raise — Qdrant accepts UUID strings as point IDs
    uuid.UUID(out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pillar1/test_vector_store.py -v`
Expected: FAIL with `ImportError: cannot import name '_to_point_id'`

- [ ] **Step 3: Add the helper and wire it into the adapter**

In `zenic/rag/vector_store.py`, add near the top (after the imports):

```python
import uuid

# Fixed namespace so a given chunk id always maps to the same Qdrant point id
# (makes re-running the migration an idempotent upsert rather than a duplicate insert).
# The literal is just any constant valid UUID — its only job is to stay fixed.
_POINT_ID_NAMESPACE = uuid.UUID("c3f1a2b4-1111-4000-8000-000000000001")


def _to_point_id(raw_id: str) -> str:
    """Map a string chunk id (e.g. 'usda_167782') to a deterministic UUID string.
    Qdrant point IDs must be an unsigned int or a UUID — not an arbitrary string."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, raw_id))
```

Then change `_QdrantAdapter.upsert` (currently around `vector_store.py:91`) from:

```python
    def upsert(self, documents: list[dict]) -> None:
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(id=d["id"], vector=d["embedding"], payload={**d["metadata"], "text": d["text"]})
            for d in documents
        ]
        self._client.upsert(collection_name=self.COLLECTION, points=points)
```

to:

```python
    def upsert(self, documents: list[dict]) -> None:
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(
                id=_to_point_id(d["id"]),
                vector=d["embedding"],
                payload={**d["metadata"], "text": d["text"], "chunk_id": d["id"]},
            )
            for d in documents
        ]
        self._client.upsert(collection_name=self.COLLECTION, points=points)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/pillar1/test_vector_store.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Add `ensure_collection` to `_QdrantAdapter`**

Add this method to `_QdrantAdapter` (place it directly above `upsert`):

```python
    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection if it does not already exist (cosine distance)."""
        from qdrant_client.models import Distance, VectorParams
        existing = {c.name for c in self._client.get_collections().collections}
        if self.COLLECTION not in existing:
            self._client.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
```

- [ ] **Step 6: Re-run the full pillar1 unit tests to confirm no import breakage**

Run: `pytest tests/pillar1/test_vector_store.py -v`
Expected: PASS (3 passed). (`ensure_collection` needs a live Qdrant, so it is exercised in Task 3, not here.)

- [ ] **Step 7: Stage and surface commit command**

```bash
git add zenic/rag/vector_store.py tests/pillar1/test_vector_store.py
```
Then tell the user to run:
```bash
git commit -m "Add Qdrant point-id helper + ensure_collection; fix upsert"
```

---

## Task 2: Migration script

**Files:**
- Create: `scripts/migrate_to_qdrant.py`

- [ ] **Step 1: Write the script**

Create `scripts/migrate_to_qdrant.py`:

```python
"""One-time migration: rebuild the production Qdrant collection from the BM25 corpus.

Run locally (NOT in the container), once, before the first deploy:

    PYTHONPATH=. python scripts/migrate_to_qdrant.py

Requires in .env (loaded automatically): QDRANT_URL, QDRANT_API_KEY, EMBED_MODEL (optional).
Embeds all 10,201 chunks with bge-small-en-v1.5 (384-dim) and upserts to Qdrant Cloud.
Idempotent: re-running upserts the same point ids rather than duplicating.
"""
import json
import os

from dotenv import load_dotenv

load_dotenv()
os.environ["ENV"] = "production"  # force the Qdrant adapter

from sentence_transformers import SentenceTransformer  # noqa: E402

from zenic.rag.vector_store import get_vector_store  # noqa: E402

CORPUS_PATH = "data/bm25_corpus.json"
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
VECTOR_SIZE = 384  # bge-small-en-v1.5
BATCH = 256


def main() -> None:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} chunks from {CORPUS_PATH}")

    model = SentenceTransformer(EMBED_MODEL)
    store = get_vector_store()
    store.ensure_collection(VECTOR_SIZE)
    print(f"Collection ready (size={VECTOR_SIZE}, cosine).")

    total = 0
    for start in range(0, len(corpus), BATCH):
        batch = corpus[start:start + BATCH]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        documents = [
            {
                "id": c["id"],
                "text": c["text"],
                "metadata": c["metadata"] if isinstance(c["metadata"], dict) else {},
                "embedding": emb.tolist(),
            }
            for c, emb in zip(batch, embeddings)
        ]
        store.upsert(documents)
        total += len(documents)
        print(f"  upserted {total}/{len(corpus)}")

    print(f"Done. Upserted {total} chunks to Qdrant collection 'zenic_knowledge'.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports cleanly (no run yet — that is Task 3)**

Run: `python -c "import ast; ast.parse(open('scripts/migrate_to_qdrant.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Stage and surface commit command**

```bash
git add scripts/migrate_to_qdrant.py
```
Then tell the user to run:
```bash
git commit -m "Add Qdrant migration script"
```

---

## Task 3: Run the migration (CHECKPOINT — writes to the real Qdrant cluster)

**Files:** none (operational step)

> This task mutates the user's real Qdrant Cloud cluster. PAUSE for the user to
> confirm `QDRANT_URL` + `QDRANT_API_KEY` are set in `.env` before running.

- [ ] **Step 1: Confirm credentials are present**

Run: `python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('URL set:', bool(os.getenv('QDRANT_URL'))); print('KEY set:', bool(os.getenv('QDRANT_API_KEY')))"`
Expected: `URL set: True` and `KEY set: True`

- [ ] **Step 2: Run the migration**

Run: `PYTHONPATH=. python scripts/migrate_to_qdrant.py`
Expected: progress lines ending in `Done. Upserted 10201 chunks ...` (count may differ slightly if the corpus changes).

- [ ] **Step 3: Verify the collection count in Qdrant**

Run:
```bash
python -c "from dotenv import load_dotenv; load_dotenv(); import os; from qdrant_client import QdrantClient; c=QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY']); print(c.count(collection_name='zenic_knowledge'))"
```
Expected: `count=10201` (matches the migration output).

- [ ] **Step 4: Smoke-test a production retrieval locally**

Run:
```bash
ENV=production PYTHONPATH=. python -c "from zenic.rag.pipeline import _try_load_bm25_from_disk, retrieve; _try_load_bm25_from_disk(); print(retrieve('how much protein per day for muscle gain')[:1])"
```
Expected: at least one chunk returned with text + metadata (proves the production Qdrant path works end-to-end on your machine).

---

## Task 4: Dockerfile

**Files:**
- Create: `Dockerfile`

- [ ] **Step 1: Write the Dockerfile**

Create `Dockerfile`:

```dockerfile
FROM python:3.12-slim

# HF Spaces runs the container as UID 1000 — create that user up front.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    ENV=production \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# Install CPU-only torch first (smaller image), then the rest of the deps.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-bake the embedding + reranker models into the image (no cold-start download).
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"

# App code + the BM25 corpus (loaded at startup).
COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "zenic/ui/app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
```

- [ ] **Step 2: Stage and surface commit command**

```bash
git add Dockerfile
```
Then tell the user to run:
```bash
git commit -m "Add HF Spaces Dockerfile (UID 1000, port 7860, pre-baked models)"
```

---

## Task 5: .dockerignore

**Files:**
- Create: `.dockerignore`

- [ ] **Step 1: Write .dockerignore**

Create `.dockerignore`:

```gitignore
.git
.github
chroma_db/
tests/
docs/
scripts/test_*.py
scripts/debug_*.py
debug_*.py
eval_data/
eval_results/
ragas_results/
OVERVIEW/
TESTING/
.playwright-mcp/
*.png
tmp_*.txt
summary.txt
NEXT_SESSION.md
.pytest_cache/
__pycache__/
*.pyc
.env
.env.*
!.env.example
.claude/
.gemini/
*.code-workspace
```

- [ ] **Step 2: Verify bm25 corpus is NOT excluded**

Run: `python -c "print('data/' not in open('.dockerignore').read())"`
Expected: `True` (we do NOT ignore `data/`, so `data/bm25_corpus.json` ships in the image).

- [ ] **Step 3: Stage and surface commit command**

```bash
git add .dockerignore
```
Then tell the user to run:
```bash
git commit -m "Add .dockerignore to keep build context small"
```

---

## Task 6: README YAML front-matter + .env.example fix

**Files:**
- Modify: `README.md`
- Modify: `.env.example`

- [ ] **Step 1: Prepend HF Space YAML front-matter to README.md**

Add this as the very first lines of `README.md` (before the existing content):

```markdown
---
title: Zenic
emoji: 🥗
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

```

- [ ] **Step 2: Fix the stale comment in .env.example**

In `.env.example`, replace line 17:

```
# REMINDER: fill these in before deploying to Streamlit Community Cloud
```
with:
```
# REMINDER: set these as Space secrets before deploying to Hugging Face Spaces
```

- [ ] **Step 3: Stage and surface commit command**

```bash
git add README.md .env.example
```
Then tell the user to run:
```bash
git commit -m "Add HF Space README metadata; fix stale env comment"
```

---

## Task 7: Local Podman smoke test (CHECKPOINT)

**Files:** none (operational step)

- [ ] **Step 1: Build the image**

Run: `podman build -t zenic .`
Expected: build completes; the model-prebake `RUN` line downloads bge-small + bge-reranker-base once during the build.

- [ ] **Step 2: Run the container against Qdrant Cloud**

Run: `podman run --rm -p 7860:7860 --env-file .env zenic`
Expected: Streamlit logs `You can now view your Streamlit app ... :7860`.

- [ ] **Step 2b: Confirm .env has ENV=production for the smoke test**

The container must read from Qdrant Cloud, not local Chroma. Ensure `.env` contains `ENV=production` (the Dockerfile also sets it, but `--env-file` overrides). If `.env` has `ENV=development`, the smoke test will hit a non-existent local Chroma — set it to `production` for this test.

- [ ] **Step 3: Manually smoke-test all 6 intents in the browser**

Open `http://localhost:7860` and exercise one prompt per intent (RAG nutrition question, RAG exercise question, TDEE/calorie calc, meal/workout plan, trend/insight, and a safety-trigger message). Confirm answers return, cite sources, and no errors appear in the container logs.
Expected: all 6 respond correctly. If any fail, STOP and debug before deploying.

---

## Task 8: Deploy to HF Spaces (CHECKPOINT)

**Files:** none (operational step)

> Requires the user's HF account + the chosen Space name (e.g. `ripun/zenic`).

- [ ] **Step 1: Create the Space (user action)**

In the HF UI: New Space → Owner = your username → Space name (e.g. `zenic`) → SDK = **Docker** → Blank → Create. Note the git URL: `https://huggingface.co/spaces/<user>/zenic`.

- [ ] **Step 2: Set Space secrets (user action)**

In Space → Settings → Variables and secrets, add: `GROQ_API_KEY`, `USDA_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `GOOGLE_API_KEY`, and `ENV=production`.

- [ ] **Step 3: Add the HF remote and push**

Run:
```bash
git remote add hf https://huggingface.co/spaces/<user>/zenic
git push hf main
```
(If the Space repo already has a README commit, use `git push hf main --force` for this first push.)
Expected: HF "Building" status appears in the Space UI; the build runs the Dockerfile (including the model prebake).

- [ ] **Step 4: Smoke-test the live URL**

Open `https://huggingface.co/spaces/<user>/zenic`. Exercise the same 6 intents as Task 7 Step 3.
Expected: all 6 respond correctly against Qdrant Cloud. This live URL is the deployed demo to review before the production-grade cycle.

---

## Notes for the executor

- The migration (Task 3) and both smoke tests (Tasks 7–8) are operational checkpoints requiring the user's real credentials and judgment — pause for the user at each.
- Commits: `git add` the files, then surface the `git commit` command for the user to run. Do not run `git commit` yourself.
- If the HF build hits a timeout or image-size limit during the model prebake, that is a deploy-environment constraint to report to the user, not a code bug — note it and stop.
