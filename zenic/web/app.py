"""Same-origin browser API with short-lived, private in-memory sessions."""
from __future__ import annotations

import asyncio
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from zenic.agent.turn import extract_reply, run_turn
from zenic.config import get_settings
from zenic.errors import ConfigError
from zenic.logging_config import get_logger

logger = get_logger(__name__)
STATIC = Path(__file__).parent / "static"
SESSION_SECONDS = 30 * 60
MAX_SESSIONS = 100
MAX_BODY_BYTES = 5000
MAX_ACTIVE_TURNS = 2


@dataclass
class Session:
    messages: list[dict[str, str]] = field(default_factory=list)
    profile: dict = field(default_factory=dict)
    pending_intent: str | None = None
    pending_missing_fields: list[str] = field(default_factory=list)
    pdf_path: Path | None = None
    touched: float = field(default_factory=time.monotonic)
    busy: bool = False


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get(self, sid: str | None, *, create: bool = False) -> tuple[str | None, Session | None]:
        with self._lock:
            now = time.monotonic()
            for key, value in list(self._sessions.items()):
                if not value.busy and now - value.touched > SESSION_SECONDS:
                    _remove_pdf(value.pdf_path)
                    del self._sessions[key]
            if sid and sid in self._sessions:
                session = self._sessions[sid]
                session.touched = now
                return sid, session
            if not create:
                return None, None
            if len(self._sessions) >= MAX_SESSIONS:
                inactive = [(key, item) for key, item in self._sessions.items() if not item.busy]
                if not inactive:
                    return None, None
                oldest_key, oldest = min(inactive, key=lambda pair: pair[1].touched)
                _remove_pdf(oldest.pdf_path)
                del self._sessions[oldest_key]
            sid = secrets.token_urlsafe(32)
            session = Session()
            self._sessions[sid] = session
            return sid, session

    def claim(self, session: Session) -> bool:
        with self._lock:
            if session.busy:
                return False
            session.busy = True
            return True

    def release(self, session: Session) -> None:
        with self._lock:
            session.busy = False
            session.touched = time.monotonic()


def _remove_pdf(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
        path.parent.rmdir()
    except OSError:
        logger.warning("could not remove expired PDF")


store = SessionStore()
active_turns = threading.BoundedSemaphore(MAX_ACTIVE_TURNS)


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: dict) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


def _api_response(payload: dict, status: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status, headers={"Cache-Control": "no-store"})


def _cookie(response: Response, request: Request, sid: str) -> None:
    response.set_cookie(
        "zenic_session", sid, max_age=SESSION_SECONDS, httponly=True,
        secure=request.url.scheme == "https", samesite="strict", path="/",
    )


async def home(_: Request) -> Response:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


async def health(_: Request) -> Response:
    try:
        get_settings().require_groq_api_key()
    except ConfigError:
        return _api_response({"status": "unconfigured"}, 503)
    return _api_response({"status": "ok"})


async def current_session(request: Request) -> Response:
    _, session = store.get(request.cookies.get("zenic_session"))
    if session is None:
        return _api_response({"messages": [], "profile": {}, "download": False})
    return _api_response({
        "messages": session.messages, "profile": session.profile,
        "download": bool(session.pdf_path and session.pdf_path.is_file()),
    })


async def reset_session(request: Request) -> Response:
    _, session = store.get(request.cookies.get("zenic_session"))
    if session is None:
        return _api_response({"ok": True})
    if not store.claim(session):
        return _api_response({"error": "A response is still in progress."}, 409)
    try:
        _remove_pdf(session.pdf_path)
        session.messages.clear()
        session.profile.clear()
        session.pending_intent = None
        session.pending_missing_fields.clear()
        session.pdf_path = None
    finally:
        store.release(session)
    return _api_response({"ok": True})


async def download(request: Request) -> Response:
    _, session = store.get(request.cookies.get("zenic_session"))
    if session is None or session.pdf_path is None or not session.pdf_path.is_file():
        return _api_response({"error": "No plan is available in this session."}, 404)
    return FileResponse(
        session.pdf_path, media_type="application/pdf", filename="zenic_health_plan.pdf",
        headers={"Cache-Control": "no-store"},
    )


async def chat(request: Request) -> Response:
    if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
        return _api_response({"error": "Expected JSON."}, 415)
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BODY_BYTES:
            return _api_response({"error": "Message is too long."}, 413)
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return _api_response({"error": "Invalid JSON."}, 400)
    prompt = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
        return _api_response({"error": "Enter a message of at most 4,000 characters."}, 422)
    try:
        get_settings().require_groq_api_key()
    except ConfigError:
        return _api_response({"error": "Zenic is not configured. Add GROQ_API_KEY."}, 503)

    if not active_turns.acquire(blocking=False):
        return _api_response({"error": "The demo is busy. Please try again soon."}, 503)
    sid, session = store.get(request.cookies.get("zenic_session"), create=True)
    if session is None:
        active_turns.release()
        return _api_response({"error": "The demo is busy. Please try again soon."}, 503)
    if not store.claim(session):
        active_turns.release()
        return _api_response({"error": "A response is still in progress."}, 409)

    async def events():
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(event: dict | None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def work() -> None:
            user_message = {"role": "user", "content": prompt.strip()}
            try:
                final, error, timings = run_turn(
                    [*session.messages, user_message], session.profile,
                    pending_intent=session.pending_intent,
                    pending_missing_fields=session.pending_missing_fields,
                    on_stage=lambda label, elapsed: emit({
                        "type": "stage", "label": label, "elapsed_ms": elapsed,
                    }),
                )
                if error:
                    emit({"type": "error", "message": error})
                    return
                assert final is not None
                reply = extract_reply(final)
                metrics = final.get("tool_results") if final.get("intent") == "calculate" else None
                assistant = {"role": "assistant", "content": reply}
                if metrics:
                    assistant["metrics"] = {
                        key: metrics[key] for key in ("tdee", "bmr", "protein_g") if key in metrics
                    }
                session.messages = [*session.messages, user_message, assistant][-40:]
                session.profile = final.get("user_profile") or session.profile
                session.pending_intent = final.get("intent") if final.get("awaiting_input") else None
                session.pending_missing_fields = (
                    final.get("missing_fields") or [] if final.get("awaiting_input") else []
                )
                new_pdf = (final.get("tool_results") or {}).get("pdf_path")
                if new_pdf:
                    old_pdf = session.pdf_path
                    session.pdf_path = Path(new_pdf)
                    if old_pdf != session.pdf_path:
                        _remove_pdf(old_pdf)
                emit({
                    "type": "final", "message": assistant, "profile": session.profile,
                    "download": bool(session.pdf_path), "timings_ms": timings,
                })
            except Exception as exc:
                logger.error("web turn failed", extra={"error_type": type(exc).__name__})
                emit({"type": "error", "message": "Something went wrong. Please try again."})
            finally:
                store.release(session)
                active_turns.release()
                emit(None)

        worker_task = asyncio.create_task(asyncio.to_thread(work))
        yield json.dumps({"type": "stage", "label": "Checking your message", "elapsed_ms": 0}) + "\n"
        while (event := await queue.get()) is not None:
            yield json.dumps(event, separators=(",", ":")) + "\n"
        await worker_task

    response = StreamingResponse(
        events(), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
    assert sid is not None
    _cookie(response, request, sid)
    return response


app = Starlette(routes=[
    Route("/", home),
    Route("/api/health", health),
    Route("/api/session", current_session),
    Route("/api/chat", chat, methods=["POST"]),
    Route("/api/reset", reset_session, methods=["POST"]),
    Route("/api/plan", download),
    Mount("/static", NoCacheStaticFiles(directory=STATIC), name="static"),
])


async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    )
    return response


app.add_middleware(BaseHTTPMiddleware, dispatch=security_headers)
