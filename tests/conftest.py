"""Shared fixtures.

The unit suite must run with no credentials and no network. Settings are cached
process-wide, so any test that manipulates the environment has to clear that
cache — `settings_env` does it on both sides of the test.
"""
from __future__ import annotations

import pytest

from zenic import config, http_client, llm
from zenic.rag import pipeline, vector_store


@pytest.fixture(autouse=True)
def _reset_singletons(request, monkeypatch):
    """Ensure no cached client, store, or index leaks between tests.

    The BM25 index in particular is a module global that a retrieval test can
    populate for the whole session, silently changing the results a later test
    sees.
    """
    if request.node.get_closest_marker("integration") is None:
        monkeypatch.setenv("ENV", "development")
        for key in ("GROQ_API_KEY", "USDA_API_KEY", "GOOGLE_API_KEY", "QDRANT_API_KEY", "QDRANT_URL"):
            monkeypatch.delenv(key, raising=False)
        config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()
    llm.reset_client()
    http_client.reset_client()
    vector_store.reset_vector_store()
    pipeline.reset_bm25_index()


@pytest.fixture
def settings_env(monkeypatch):
    """Set environment variables and rebuild the cached settings.

    Usage::

        def test_x(settings_env):
            settings = settings_env(ENV="development", GROQ_API_KEY="k")
    """

    def apply(**env: str):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return config.reload_settings()

    yield apply
    config.get_settings.cache_clear()
