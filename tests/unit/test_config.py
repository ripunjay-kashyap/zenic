"""Configuration loading, validation, and secret handling."""
import pytest

from zenic.errors import ConfigError


def test_defaults_are_development(settings_env, monkeypatch):
    monkeypatch.delenv("ENV", raising=False)
    settings = settings_env()
    assert settings.env == "development"
    assert settings.is_production is False
    assert settings.collection_name == "zenic_knowledge"


def test_production_requires_qdrant_credentials(settings_env, monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="QDRANT_URL"):
        settings_env(ENV="production")


def test_unknown_env_is_rejected(settings_env):
    with pytest.raises(ConfigError, match="ENV must be one of"):
        settings_env(ENV="staging")


def test_qdrant_url_must_have_a_scheme(settings_env):
    with pytest.raises(ConfigError, match="must start with http"):
        settings_env(ENV="production", QDRANT_URL="my-cluster.qdrant.io", QDRANT_API_KEY="k")


def test_top_k_cannot_exceed_the_candidate_pool(settings_env):
    with pytest.raises(ConfigError, match="RETRIEVAL_TOP_K"):
        settings_env(ENV="development", RETRIEVAL_TOP_K="50", RETRIEVAL_CANDIDATE_POOL="10")


def test_non_numeric_setting_is_rejected(settings_env):
    with pytest.raises(ConfigError, match="must be an integer"):
        settings_env(ENV="development", RETRIEVAL_TOP_K="seven")


def test_inline_comments_are_stripped(settings_env):
    """`.env` files commonly carry trailing comments on unquoted values."""
    settings = settings_env(ENV="development", GROQ_MODEL="llama-3.3-70b  # the default")
    assert settings.groq_model == "llama-3.3-70b"


def test_missing_key_raises_an_actionable_error(settings_env, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    settings = settings_env(ENV="development")
    with pytest.raises(ConfigError) as exc:
        settings.require_groq_api_key()
    assert "GROQ_API_KEY" in str(exc.value)
    assert ".env.example" in str(exc.value)


def test_redacted_never_exposes_secrets(settings_env):
    settings = settings_env(ENV="development", GROQ_API_KEY="gsk_supersecret_value")
    redacted = settings.redacted()
    assert redacted["groq_api_key"] == "<set>"
    assert "gsk_supersecret_value" not in str(redacted)


def test_redacted_marks_absent_secrets(settings_env, monkeypatch):
    monkeypatch.delenv("USDA_API_KEY", raising=False)
    settings = settings_env(ENV="development")
    assert settings.redacted()["usda_api_key"] == "<unset>"


def test_multi_query_can_be_disabled(settings_env):
    assert settings_env(ENV="development", MULTI_QUERY_ENABLED="false").multi_query_enabled is False
    assert settings_env(ENV="development", MULTI_QUERY_ENABLED="true").multi_query_enabled is True


def test_repr_never_exposes_keys(settings_env):
    settings = settings_env(ENV="development", GROQ_API_KEY="private-secret")
    assert "private-secret" not in repr(settings)


@pytest.mark.parametrize("name,value", [("HTTP_TIMEOUT_SECONDS", "nan"), ("LLM_TIMEOUT_SECONDS", "0"), ("HTTP_MAX_RETRIES", "-1"), ("MULTI_QUERY_COUNT", "100"), ("RERANK_BATCH_SIZE", "0")])
def test_invalid_resource_limits(settings_env, name, value):
    with pytest.raises(ConfigError):
        settings_env(ENV="development", **{name: value})
