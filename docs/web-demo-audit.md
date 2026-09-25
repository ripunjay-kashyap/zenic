# Browser demo audit — 2026-09-24 to 2026-09-25

This audit covers the HTML/CSS/JavaScript interface and Starlette API that replaced
Streamlit. The previous release audit records the earlier backend hardening.
The browser checks used the local four-core CPU host, production Qdrant collection,
and configured Groq models. They are functional demonstrations, not clinical
validation or a hosted-service uptime test.

## What changed

- One same-origin server now serves the browser interface, stage-streaming chat API,
  short-lived server-side sessions, and session-scoped PDF downloads. The UI needs
  no JavaScript build tool or external CDN. Two active graph turns are allowed
  per process; excess requests receive a clear busy response.
- The retrieval candidate pool fell from 30 to 12 after a live relevance check.
  The cross encoder still reranks every selected candidate in short,
  length-sorted batches. Referential follow-ups are rewritten into standalone
  questions before retrieval.
- Factual generation includes a source title and validated NIH URL. Age-specific
  intake questions use the matching NIH table row for relevance, and the
  synthetic upper-limit summary is excluded from recommended-intake searches.
  The full source table remains available to the answer generator.
- The production container uses a separate 85-package lock without local Chroma,
  ingestion, evaluation, or test dependencies. The development lock retains those
  tools for reproducibility.

## Browser and answer checks

Playwright Chromium checked desktop (1440 × 900) and mobile (390 × 844) layouts,
responsive sidebar, session restoration, synthetic weekly PDF download, and a
two-turn calculation flow with profile updates. The final browser regression
answered these source-specific questions and rendered a clickable NIH citation:

| Question | Observed answer | Turn time |
| --- | --- | ---: |
| NIH vitamin D recommended intake, ages 19–70 | 15 mcg (600 IU), cited NIH fact sheet | 17.0 s |
| “That same fact sheet” for adults older than 70 | 20 mcg (800 IU), cited the same age table | 10.5 s |

Both values match the [NIH Office of Dietary Supplements vitamin D table](https://ods.od.nih.gov/factsheets/VitaminD-HealthProfessional/).
An earlier run exposed a wrong 600–800 IU range for ages 19–70 and an upper-limit
answer for the older-adult follow-up. The retrieval and generation corrections
above were made before the final browser regression. Chromium reported no console
errors in that run.

The browser also completed the synthetic weekly summary in about 1.2 seconds,
the two-turn TDEE calculation in about 1.8 seconds for the final turn, and a
separate cited ISSN protein question in 17.2 seconds. These times are individual
observations, not latency percentiles. A separate direct provider run took
28 seconds for the older-adult query when Qdrant was slower. Query time still
depends on network latency, source retrieval, and CPU reranking. The UI streams
stage labels and elapsed time so a slow turn remains legible.

## Checks

- Offline regression suite: **246 passed**, 24 live integration cases deselected.
- Live retrieval spot check after the ranking change: **10 passed, 2 intentionally
  skipped** (the API-routing cases use a separate script).
- Live RAG-versus-USDA routing sweep: **6/6 passed**. Four indexed questions used
  RAG without API fallback; two absent-food questions abstained. No successful
  USDA API fallback is demonstrated by these cases.
- Ruff, JavaScript syntax, dependency compatibility, wheel static-asset packaging,
  desktop/mobile Chromium render, and browser console checks passed.
- The full development lock has nine known advisory entries across three packages
  without fixes. The production-only lock has **no known findings** in the current
  `pip-audit` database; separately installed CPU PyTorch 2.13.0 also had no
  known package-version findings. [SECURITY.md](../SECURITY.md) records the remaining local
  development restrictions.
- Production container `zenic-web-audit:20260924` built successfully (image
  `48042c92328a`). Its `pip check`, HTTP health/page/static/session endpoints,
  and one live cited vitamin D RAG turn passed; the final-image turn took 29.3 s.
  Runtime inspection confirmed UID 1000, CPU PyTorch, the bundled corpus, no
  copied `.env`, and no ChromaDB, RAGAS, or diskcache installation.

The server has no account authentication. A public live demo needs an authenticated
gateway, TLS, request limits, provider budget controls, private ephemeral storage,
and a retention policy. It has not been deployed publicly in this audit.
