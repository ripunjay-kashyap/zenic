"""Retry, backoff, and error translation in the shared HTTP client."""
import httpx
import pytest

from zenic import http_client
from zenic.errors import ExternalAPIError


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Retries are exercised without the real backoff delay."""
    monkeypatch.setattr(http_client.time, "sleep", lambda _seconds: None)


def _install(monkeypatch, responses):
    """Serve `responses` (status codes or exceptions) in order."""
    calls = {"count": 0}

    class _Client:
        def get(self, url, params=None, timeout=None):
            item = responses[min(calls["count"], len(responses) - 1)]
            calls["count"] += 1
            if isinstance(item, Exception):
                raise item
            request = httpx.Request("GET", url)
            return httpx.Response(item, json={"ok": True}, request=request)

    monkeypatch.setattr(http_client, "get_client", lambda: _Client())
    return calls


def test_successful_request_returns_the_body(monkeypatch):
    _install(monkeypatch, [200])
    assert http_client.get_json("https://example.test", service="Test") == {"ok": True}


def test_server_error_is_retried_then_succeeds(monkeypatch):
    calls = _install(monkeypatch, [503, 200])
    assert http_client.get_json("https://example.test", service="Test") == {"ok": True}
    assert calls["count"] == 2


def test_rate_limit_is_retried(monkeypatch):
    calls = _install(monkeypatch, [429, 200])
    http_client.get_json("https://example.test", service="Test")
    assert calls["count"] == 2


def test_connection_error_is_retried(monkeypatch):
    calls = _install(monkeypatch, [httpx.ConnectError("refused"), 200])
    http_client.get_json("https://example.test", service="Test")
    assert calls["count"] == 2


def test_client_error_is_not_retried(monkeypatch):
    """A 404 will never succeed on retry — fail immediately."""
    calls = _install(monkeypatch, [404])
    with pytest.raises(ExternalAPIError, match="HTTP 404"):
        http_client.get_json("https://example.test", service="Test")
    assert calls["count"] == 1


def test_exhausted_retries_raise_with_the_service_name(monkeypatch):
    _install(monkeypatch, [503])
    with pytest.raises(ExternalAPIError, match="USDA is unreachable"):
        http_client.get_json("https://example.test", service="USDA")


def test_non_json_body_is_reported_clearly(monkeypatch):
    class _Client:
        def get(self, url, params=None, timeout=None):
            return httpx.Response(200, text="<html>oops</html>", request=httpx.Request("GET", url))

    monkeypatch.setattr(http_client, "get_client", lambda: _Client())
    with pytest.raises(ExternalAPIError, match="non-JSON"):
        http_client.get_json("https://example.test", service="Test")


def test_json_array_is_rejected(monkeypatch):
    class _Client:
        def get(self, url, params=None, timeout=None):
            return httpx.Response(200, json=[1, 2, 3], request=httpx.Request("GET", url))

    monkeypatch.setattr(http_client, "get_client", lambda: _Client())
    with pytest.raises(ExternalAPIError, match="expected a JSON object"):
        http_client.get_json("https://example.test", service="Test")


def test_default_timeout_is_not_disabled(monkeypatch, settings_env):
    settings_env(ENV="development", HTTP_TIMEOUT_SECONDS="7")
    seen = []
    class Client:
        def get(self, url, params=None, timeout=None):
            seen.append(timeout)
            return httpx.Response(200, json={}, request=httpx.Request("GET", url))
    monkeypatch.setattr(http_client, "get_client", lambda: Client())
    http_client.get_json("https://example.test", service="Test")
    assert seen == [7]
