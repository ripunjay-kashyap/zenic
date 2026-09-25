"""The browser API must keep conversations and generated files session scoped."""
from __future__ import annotations

import importlib
import json

from starlette.testclient import TestClient

web = importlib.import_module("zenic.web.app")


class Configured:
    def require_groq_api_key(self):
        return "available"


def _fake_turn(messages, profile, *, on_stage, **_kwargs):
    on_stage("Writing the answer", 12)
    return ({
        "messages": [*messages, {"role": "assistant", "content": "Grounded answer [1]."}],
        "intent": "nutrition_qa", "user_profile": profile,
        "awaiting_input": False, "tool_results": {},
    }, None, {"router": 12, "total": 18})


def test_chat_stream_session_and_reset(monkeypatch):
    monkeypatch.setattr(web, "get_settings", Configured)
    monkeypatch.setattr(web, "run_turn", _fake_turn)
    monkeypatch.setattr(web, "store", web.SessionStore())
    with TestClient(web.app) as client:
        response = client.post("/api/chat", json={"message": "What is protein?"})
        assert response.status_code == 200
        assert "HttpOnly" in response.headers["set-cookie"]
        assert "SameSite=strict" in response.headers["set-cookie"]
        assert response.headers["cache-control"] == "no-store"
        events = [json.loads(line) for line in response.text.splitlines()]
        assert [event["type"] for event in events] == ["stage", "stage", "final"]
        assert events[-1]["message"]["content"] == "Grounded answer [1]."
        assert events[-1]["timings_ms"]["total"] == 18
        assert len(client.get("/api/session").json()["messages"]) == 2

        other = TestClient(web.app)
        assert other.get("/api/session").json()["messages"] == []
        assert other.get("/api/plan").status_code == 404

        assert client.post("/api/reset").json() == {"ok": True}
        assert client.get("/api/session").json()["messages"] == []


def test_input_bounds_and_content_type(monkeypatch):
    monkeypatch.setattr(web, "get_settings", Configured)
    with TestClient(web.app) as client:
        assert client.post("/api/chat", content="message=hi").status_code == 415
        assert client.post("/api/chat", content=b"{", headers={"Content-Type": "application/json"}).status_code == 400
        assert client.post("/api/chat", json={"message": " "}).status_code == 422
        assert client.post("/api/chat", json={"message": "x" * 5001}).status_code == 413
        assert client.get("/").headers["content-security-policy"].startswith("default-src 'self'")
        assert client.get("/static/app.js").headers["cache-control"] == "no-cache"


def test_busy_demo_rejects_extra_turn_without_storing_message(monkeypatch):
    class Full:
        def acquire(self, **_kwargs):
            return False

    monkeypatch.setattr(web, "get_settings", Configured)
    monkeypatch.setattr(web, "store", web.SessionStore())
    monkeypatch.setattr(web, "active_turns", Full())
    with TestClient(web.app) as client:
        assert client.post("/api/chat", json={"message": "hello"}).status_code == 503
        assert client.get("/api/session").json()["messages"] == []


def test_pdf_only_available_in_own_session(monkeypatch, tmp_path):
    path = tmp_path / "plan.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    def with_pdf(messages, profile, **_kwargs):
        return ({
            "messages": [*messages, {"role": "assistant", "content": "Your plan is ready."}],
            "intent": "weekly_summary", "user_profile": profile,
            "awaiting_input": False, "tool_results": {"pdf_path": str(path)},
        }, None, {"total": 1})

    monkeypatch.setattr(web, "get_settings", Configured)
    monkeypatch.setattr(web, "run_turn", with_pdf)
    monkeypatch.setattr(web, "store", web.SessionStore())
    with TestClient(web.app) as owner, TestClient(web.app) as stranger:
        owner.post("/api/chat", json={"message": "Show my demo report"})
        assert owner.get("/api/plan").content.startswith(b"%PDF")
        assert stranger.get("/api/plan").status_code == 404
        owner.post("/api/reset")
        assert owner.get("/api/plan").status_code == 404
        assert not path.exists()
