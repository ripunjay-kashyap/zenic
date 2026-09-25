# Release audit — 2026-09-24

This records the original Streamlit release before the browser UI migration.
See [the current web demo audit](web-demo-audit.md) for the new interface and
latency measurements.

## Decision

The original Qdrant connectivity blocker was resolved on September 24 after the
operator supplied a new cluster. It reports a green collection with all 10,201
passages; a complete remote scan verified IDs, text, metadata, and source counts
against the bundled corpus. All release checks below passed. The release can be
published by fast-forwarding GitHub's existing `main` branch.

The initial Groq model, `llama-3.3-70b-versatile`, returned HTTP 404 and was absent
from the account's model list. The application default and example configuration
now use `openai/gpt-oss-20b` for chat and routing. Complex plans use
`openai/gpt-oss-120b`, selected after the smaller model repeatedly produced
malformed nested meal-plan JSON. Both are available to the configured account.
Explicit model overrides must select available models. The operator replaced
Qdrant credentials locally; no credentials were added to the repository.

## Changes reviewed

The starting tree already contained a large staged deployment/hardening pass.
This audit retained that implementation and corrected material issues:

- Factual generation no longer switches to ungrounded chat when retrieval fails.
  It requires relevant evidence, bounded whole-passage context, and valid source
  IDs. Live USDA fallback results pass through the same reranker.
- Document instructions remain untrusted data; history cannot inject privileged
  message roles. Inputs, history, context, retries, and model output are bounded.
- Stable IDs align vector and lexical candidates. Reciprocal rank fusion replaces
  the incompatible raw-score sum, and fair source allocation prevents common
  sources from filling the pool before USDA candidates are considered. BM25 uses partial top-k
  selection rather than sorting every corpus score. Boilerplate is removed
  before candidate limits. An ISSN chunking loop that could fail to terminate
  and oversized dietary-guideline chunks were corrected.
- The shared HTTP helper now preserves its configured timeout and uses one retry
  budget. Secrets are excluded from settings repr; environment overrides win
  over local dotenv values. Production Qdrant requires HTTPS.
- The local ignored `.env` is now owner-readable only (mode 600). Logs omit
  health queries, profile measurements, raw provider responses, and
  exception bodies. SDK debug logging is suppressed. Generated PDFs live in
  private temporary directories; previous session files are removed on replacement.
- Calculations reject unsupported child profiles and inferred physiology
  coefficients. Weekly reports explicitly identify synthetic demonstration data.
  The standalone OpenFDA utility no longer labels a substance safe from event counts.
- Plans require source material, typed fields, finite nonnegative nutrition
  estimates, and seven complete meal-plan days. Generation uses constrained
  schemas on supported models and validates the result locally. Invalid JSON
  gets one corrective retry; unrelated provider errors are not retried by that
  mechanism. Plan output remains bounded with low reasoning effort.
- HTTP 413 during factual generation reduces the evidence set from seven to
  three to one whole passage, with at most three generation attempts. Citation
  IDs are checked against the actual reduced context. This handles provider
  request limits without cutting passages or answering without evidence.
- Dependency resolution replaces the incompatible legacy Google SDK combination,
  upgrades available security fixes, and targets Python 3.12 plus pinned CPU Torch.
  CI installs the resolved lock and checks dependency consistency.
- Superseded planning notes, unsupported probes, debug artifacts, and stale
  screenshots were removed. Corpus-provenance scripts and historical evaluation
  data remain explicitly labeled. A corpus indexing CLI makes fresh clones usable.
- Migration dry-run now exits before any collection/index creation. Corpus JSON
  persistence uses atomic replacement to avoid publishing a partial file.

## Validation evidence

- Offline regression suite: **231 passed**, 24 integration cases deselected.
- Ruff and Python bytecode compilation: passed.
- Installed dependency consistency (`uv pip check`): passed.
- Container image build: passed (`zenic-audit:20260924`, image `7e47723669a1`).
  Network-disabled container smoke checks confirmed UID 1000, the bundled corpus,
  no `.env` or Streamlit secrets in `/home/user/app`, and a clean `pip check` result.
- Streamlit AppTest: initial page loads without exceptions.
- Corpus: all 10,201 documents have valid text/metadata and unique IDs.
- Real local embedding, Chroma/BM25 search, and reranking: passed.
- Production collection: all 10,201 points match the bundled corpus. Migration
  reused 10,157 local vectors with exact ID/text matches after 15 fresh embedding
  comparisons across all five sources (cosine similarity above 0.99999). The 44
  changed passages were embedded again. The committed indexing CLI reproduces
  all vectors from the bundled corpus when no local cache is available.
- Production readiness: **8/8 checks passed**, including both configured models,
  the collection, and indexed source filtering.
- Final production workflow sweep: **7/7 passed** across cited nutrition Q&A,
  missing-evidence abstention, greeting, adult calculation, demo weekly PDF,
  workout PDF, and seven-day meal PDF. All PDFs were opened and checked for a
  valid PDF header. No personal health data was used.
- RAG-versus-API routing spot check: **6/6 passed**. Four indexed questions used
  RAG without API fallback; two absent-food questions abstained when no relevant
  API evidence was available. Neither received an ungrounded answer. This check
  does not demonstrate a successful USDA API fallback response.
- Live Groq/local-corpus graph turns: cited vitamin-D and protein answers,
  fictional-health abstention, greeting, adult calculation, synthetic weekly
  summary, workout plan, and seven-day meal plan. All three document workflows
  produced PDF files. These are functional checks, not clinical validation.
- One supported-answer generation abstained under strict citation validation;
  a repeated call returned valid citations. Grouped numeric citations now have
  explicit regression coverage. Model output remains nondeterministic.
- Live integration suite: **22 passed, 2 intentionally skipped** (the existing
  API-enforcement cases handled by a separate script). The
  original plant-food case required USDA for a general food-list question. Its
  source check now permits authoritative NIH, ISSN, and dietary-guideline passages
  and additionally requires two named plant foods. USDA-only assertions remain
  for nutrient-quantity questions. No test was disabled to hide the failure.
- Migration `--dry-run`: reads and validates the corpus without contacting or
  mutating Qdrant.
- Gitleaks 8.30.0: no leaks in all 13 existing commits, application source, or an
  intended-file snapshot including the bundled corpus. Local `.env` was excluded.
- Dependency audit: available fixes reduced findings from 40 to 9 across three
  packages. The installed environment was also audited; its CPU Torch version
  suffix was unrecognized, so upstream Torch 2.13.0 was audited separately with
  no known findings. No-fixed-version findings and exposure restrictions are recorded in
  [SECURITY.md](../SECURITY.md); this is not a zero-vulnerability result.

A hosted UI deployment has not been performed; validation uses the local app
against production Qdrant and Groq, plus the built container.
Historical RAGAS scores were not rerun and do not describe these changed prompts.

## Git state

- Release branch: `ship-hf-spaces`; pre-release HEAD `fd46664`.
- Remote: `https://github.com/ripunjkashyap-a11y/zenic.git`.
- Before publication, remote's only branch was `main` at
  `b7766f1901bd8d3b84ae8cbb0ca37862e55fae92`.
- At audit start, four local commits on `ship-hf-spaces` followed remote `main`.
  They are all in the fast-forward ancestry; no remote history needs to be
  overwritten.
- No unresolved merge entries or working-tree diff whitespace errors were found.
- GitHub's default branch is `main`. Both local and remote `main` are ancestors
  of the reviewed release branch; a normal fast-forward preserves the history.

## Remaining operating limits

Authentication, distributed request/concurrency limiting, private ephemeral disk,
provider budgets, and log retention must be supplied by the deployment. Prompt
rules and citation-ID validation cannot guarantee clinical correctness or resist
all injection attempts. The small synthetic test set is not a medical evaluation.
Source content retains its source licensing; the repository's MIT license covers
project code, not a relicensing of third-party documents.

The selected free Qdrant cluster has no production availability guarantees and
may suspend or be deleted after inactivity. Reindexing is reproducible, but a
durable public service needs an appropriate hosting plan and backups. See
[Qdrant's cluster documentation](https://qdrant.tech/documentation/cloud/create-cluster/).
