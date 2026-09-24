# Corpus provenance

These three historical scripts record the manually authored summary chunks in
`data/bm25_corpus.json`. They are retained for auditability, not as a supported
production ingestion workflow. Their metadata marks the chunks as synthetic.
Do not rerun them against a production store. Prefer source extraction through
`scripts/run_ingestion.py`; inspect the original document before replacing a summary.
