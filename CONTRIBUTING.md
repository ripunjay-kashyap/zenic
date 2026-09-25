# Contributing

Use Python 3.12 and the setup steps in [README.md](README.md). Never commit `.env`,
private health data, local databases, model weights, or generated outputs.

## Validation

```bash
ruff check .
pytest -m 'not integration' -q
python -m compileall -q zenic scripts
pip check
```

Integration tests require a populated vector store and live provider credentials.
Run them explicitly with `pytest -m integration`. Use synthetic health questions
for manual tests. Record provider/model versions, retrieval settings, outcomes,
and limitations rather than publishing scores without context.

## Dependency updates

`requirements.txt` describes direct dependencies and security minimums.
`requirements.lock.txt` is the full resolved Python 3.12 package set used by CI.
`requirements.runtime.txt` and `requirements.runtime.lock.txt` are the smaller
production web set used by Docker. CPU PyTorch 2.13.0 is installed separately
from its wheel index.

To regenerate, resolve in a disposable Python 3.12 environment using
`uv pip compile requirements.txt --python-version 3.12 --output-file resolved.txt`.
Resolve the production set with
`uv pip compile requirements.runtime.txt --constraint requirements.lock.txt --python-version 3.12`.
Remove resolved `torch`, `triton`, `nvidia-*`, and `cuda-*` entries from both locks
because CPU PyTorch is supplied separately. Ensure every production package
is present at the same version in the full lock. Install each lock after the
pinned CPU wheel, run `pip check`, audit both with `pip-audit`, and run the tests.
Remove temporary resolved files. Do not use an arbitrary `pip freeze` from an
unrelated development environment.

## Code and data conventions

- Route model calls through `zenic/llm.py` and HTTP APIs through the shared client.
- Validate LLM-derived profile values before deterministic computation.
- Treat prompts, retrieved content, and generated JSON as untrusted inputs.
- Keep medical facts grounded; missing evidence must not enable unrestricted chat.
- Preserve whole numeric tables and their age/life-stage context when chunking.
- Use stable corpus IDs across BM25 and vectors. Index the same corpus in both.
- Test retrieval changes against the existing query set before changing thresholds
  or score fusion. Citation-ID checks do not prove factual entailment.
- Add focused regression tests for failure modes; avoid broad cosmetic rewrites.
- Keep logs free of health queries, measurements, credentials, and provider bodies.
