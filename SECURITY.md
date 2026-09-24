# Security

Report vulnerabilities through this repository's private GitHub security advisory
channel when enabled, rather than posting credentials or exploit details publicly.

## Trust boundaries

- Secrets come from environment configuration. `.env`, Streamlit secrets, raw
  source documents, vector databases, generated PDFs, and local agent settings
  are excluded from Git and container build context. Secret fields are omitted
  from settings repr; `Settings.redacted()` reports presence only.
- Use a Qdrant key restricted to reading the application's collection at runtime.
  Keep collection-management and write permissions in a separate operator key
  used for indexing; the chat application does not need those permissions.
- Browser sessions own their conversation/profile state. There is no account
  authentication. Public deployment requires an authenticated gateway, request
  and concurrency limits, TLS, and provider spending limits.
- Health questions and profile fields are sent to Groq; optional food searches
  go to USDA. Do not submit identifying or sensitive health information. The app
  is not designed for regulated health records.
- HTML sidebar values are escaped. Model output is rendered as Streamlit text or
  Markdown without raw HTML. Model calls cannot execute code or select file paths.
- User messages are limited to 4,000 characters; model history and completion
  sizes are bounded. HTTP calls have timeouts and a single bounded retry budget.
- Ingestion is an operator CLI, not an upload endpoint. Only ingest documents
  from reviewed sources into private, trusted filesystem paths. PDF parsing is
  not a sandbox: use a disposable, resource-limited environment for new PDFs.
- Retrieved evidence is serialized as data, explicitly separated from system
  instructions, filtered for relevance, and bounded without truncating tables.
  Missing evidence and invalid citation IDs cause abstention. These controls
  reduce prompt-injection risk; they do not guarantee semantic faithfulness.
- Safety regexes are a limited first layer. System prompts forbid diagnosis and
  prescribing, but adversarial model behavior remains possible. The OpenFDA
  utility is not part of the graph and returns `safe=None`: adverse-event counts
  cannot establish that a substance is safe.
- PDFs use private, randomly named temporary directories. Replacing a session's
  PDF deletes its previous file. Container filesystems must remain private and
  ephemeral; host-level retention cleanup is required for abandoned sessions.
- Application logs contain operation names, counts, timings, and error types,
  rather than prompts, profile measurements, provider bodies, or generated text.
  Keep provider tracing disabled and restrict access to deployment logs.

## Dependency audit — 2026-09-23

The repaired lock upgrades vulnerable GitPython, h2, pypdf, and setuptools.
`pip-audit` still reports nine findings (including duplicate advisory entries)
in three packages for which the database lists no fixed version:

| Package | Advisory IDs | Exposure and required restriction |
| --- | --- | --- |
| ChromaDB 1.5.9 | PYSEC-2026-311, PYSEC-2026-3813, PYSEC-2026-3814, PYSEC-2026-3815 | Server collection APIs permit code execution or cross-tenant access. Zenic uses embedded Chroma only in development, Qdrant in production. Never run/expose a Chroma HTTP server or enable remote model code. |
| diskcache 5.6.3 | PYSEC-2026-2447 | Pickle deserialization if an attacker can write cache files. Do not share writable cache directories with untrusted users; Zenic does not load user-supplied caches. |
| RAGAS 0.4.3 | PYSEC-2026-3046 | SSRF in multimodal evaluation context handling. RAGAS is used by an operator-only text evaluation script; never expose it as an API or evaluate untrusted multimodal contexts/URLs. |

These are documented residual risks, not clean audit results. Re-run the audit on
updates and re-evaluate applicability before enabling any affected feature.

## Rotation and incident response

Revoke exposed credentials at the provider and replace deployment secrets.
Deleting a file or commit does not revoke a credential. Rotating a Qdrant API key
does **not** erase its collection; only a new/empty cluster needs reindexing.
Do not rewrite repository history without coordinating with collaborators.

CI runs Gitleaks on repository history. Local scans must also cover the intended
working-tree files. Never print secret values in scan reports.

## References

- [OWASP RAG security](https://cheatsheetseries.owasp.org/cheatsheets/RAG_Security_Cheat_Sheet.html)
- [OWASP prompt-injection prevention](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html)
- [FDA API disclaimer](https://open.fda.gov/apis/try-the-api/)
