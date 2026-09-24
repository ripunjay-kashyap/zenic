import html
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

# Importing config loads .env (override=True, so it wins over stale shell vars)
# and validates the configuration before anything else touches it.
from zenic.config import get_settings
from zenic.errors import ConfigError, ZenicError
from zenic.logging_config import bind_correlation_id, get_logger

logger = get_logger(__name__)

from zenic.agent.graph import app as zenic_app
from zenic.agent.graph import initial_state


def esc(value) -> str:
    """Escape a value for interpolation into the raw HTML blocks below.

    Profile values originate from an LLM extraction of user-supplied text, so
    they are untrusted input; without escaping, a crafted message could inject
    markup into the sidebar.
    """
    return html.escape(str(value), quote=True)


def _greeting() -> str:
    """Time-of-day greeting for the hero."""
    hour = datetime.now().hour
    if hour < 5:
        return "Still up"
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    if hour < 22:
        return "Good evening"
    return "Burning the midnight oil"

# ---------------------------------------------------------------------------
# UI Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Zenic",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="expanded",
)

def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = os.path.join(os.path.dirname(__file__), "styles.css")
if os.path.exists(css_path):
    load_css(css_path)

# ---------------------------------------------------------------------------
# Startup checks — fail visibly, not on the first user message
# ---------------------------------------------------------------------------
try:
    _settings = get_settings()
    _settings.require_groq_api_key()
except ConfigError as exc:
    st.error(f"Zenic is not configured correctly.\n\n{exc}")
    logger.error("startup configuration check failed", extra={"error": str(exc)})
    st.stop()

logger.info("ui started", extra={"env": _settings.env, "model": _settings.groq_model})

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ---------------------------------------------------------------------------
# Helpers — Profile display
# ---------------------------------------------------------------------------
def format_profile_value(key: str, value) -> str:
    """Format a raw profile field value for sidebar display."""
    if isinstance(value, list):
        return ", ".join(format_profile_value(key, v) for v in value) if value else "—"

    if key == "weight_kg":
        return f"{value} kg"
    if key == "height_cm":
        return f"{value} cm"
    if key == "available_days":
        return f"{value} days"
    if isinstance(value, str):
        return value.replace("_", " ").title()
    return str(value)


# ---------------------------------------------------------------------------
# Sidebar — Bio-Digital Profile
# ---------------------------------------------------------------------------
with st.sidebar:
    # ── Wordmark ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="sidebar-mark">
          <div class="mark-glyph"></div>
          <span class="mark-text">Zenic</span>
          <span class="mark-sub">v.1</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    profile = st.session_state.user_profile
    REQUIRED = ["weight_kg", "height_cm", "age", "gender", "activity_level", "goal"]
    completed = [f for f in REQUIRED if profile.get(f)]
    percent = int((len(completed) / len(REQUIRED)) * 100)

    # ── Sync card ──────────────────────────────────────────────────────────
    # role/aria on the track so the completeness value is announced rather than
    # being a pair of decorative divs a screen reader skips entirely.
    st.markdown(
        f"""
        <div class="sync-card">
            <div class="sync-row">
                <span class="sync-label">Bio-Sync</span>
                <span class="sync-pct">{percent}%</span>
            </div>
            <div class="sync-track" role="progressbar"
                 aria-valuenow="{percent}" aria-valuemin="0" aria-valuemax="100"
                 aria-label="Profile completeness: {len(completed)} of {len(REQUIRED)} fields">
                <div class="sync-fill" style="width:{percent}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Profile fields ─────────────────────────────────────────────────────
    if profile:
        st.markdown("<span class='sb-label'>Biometrics</span>", unsafe_allow_html=True)

        field_icons = {
            "age": "○", "weight_kg": "◐", "height_cm": "◇", "gender": "◈",
            "activity_level": "◉", "goal": "◎", "dietary_restrictions": "◊",
            "experience_level": "◆", "available_days": "◧", "equipment": "◩",
        }
        labels = {
            "age": "Age", "weight_kg": "Weight", "height_cm": "Height", "gender": "Gender",
            "activity_level": "Activity", "goal": "Goal", "dietary_restrictions": "Diet",
            "experience_level": "Experience", "available_days": "Schedule", "equipment": "Gym",
        }

        rows_html = []
        for key, icon in field_icons.items():
            val = profile.get(key)
            if val is not None and val != "":
                display_val = esc(format_profile_value(key, val))
                rows_html.append(
                    f"""<div class="field-row">
                        <span class="field-icon">{icon}</span>
                        <div class="field-body">
                            <span class="field-label">{esc(labels.get(key, key.title()))}</span>
                            <span class="field-value">{display_val}</span>
                        </div>
                    </div>"""
                )
        st.markdown("\n".join(rows_html), unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div class="profile-empty">
                <span class="glow-pip"></span>
                Awaiting first signal &mdash; start a conversation to build your biometric profile.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── PDF vault ──────────────────────────────────────────────────────────
    if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
        st.markdown("<span class='sb-label'>Vault</span>", unsafe_allow_html=True)
        with open(st.session_state.pdf_path, "rb") as f:
            st.download_button(
                "Download Plan",
                f,
                file_name="zenic_health_plan.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def render_metric_cards(results):
    if not results:
        return
    cols = st.columns(3)
    if "tdee" in results:
        with cols[0]:
            st.markdown(
                f"""<div class='metric-card'>
                    <span class='metric-label'>TDEE</span>
                    <span class='metric-value'>{int(results['tdee'])}</span>
                    <span class='metric-sub'>KCAL / DAY</span>
                </div>""",
                unsafe_allow_html=True,
            )
    if "bmr" in results:
        with cols[1]:
            st.markdown(
                f"""<div class='metric-card'>
                    <span class='metric-label'>BMR</span>
                    <span class='metric-value'>{int(results['bmr'])}</span>
                    <span class='metric-sub'>KCAL / DAY</span>
                </div>""",
                unsafe_allow_html=True,
            )
    if "protein_g" in results:
        with cols[2]:
            st.markdown(
                f"""<div class='metric-card'>
                    <span class='metric-label'>PROTEIN</span>
                    <span class='metric-value'>{int(results['protein_g'])}g</span>
                    <span class='metric-sub'>DAILY TARGET</span>
                </div>""",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Chat Engine
# ---------------------------------------------------------------------------
if not st.session_state.messages and not st.session_state.pending_prompt:
    # ── Hero Section ──────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="hero">
          <div class="orb-wrap">
            <div class="orb-glow"></div>
            <div class="orb"></div>
          </div>
          <div class="hero-greeting">
            {_greeting()}<span class="accent">.</span>
          </div>
          <p class="hero-sub">
            What shall we synthesise today &mdash; nutrition, training, or a precision plan?
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Entry Points ───────────────────────────────────────────────────────
    st.markdown(
        "<p class='prompt-section-label'>// SELECT ENTRY POINT</p>",
        unsafe_allow_html=True,
    )

    SAMPLE_PROMPTS = [
        ("🔬", "What does the ISSN recommend for protein intake for athletes?"),
        ("⚡", "Calculate my TDEE — I'm 28, 75 kg, 178 cm, moderately active"),
        ("📋", "Give me a 7-day meal plan for muscle gain"),
        ("💪", "Best barbell exercises for back hypertrophy?"),
        ("📊", "How has my calorie intake trended this week?"),
        ("🥗", "What supplements does the NIH recommend for endurance athletes?"),
    ]

    col1, col2 = st.columns(2)
    for i, (icon, prompt_text) in enumerate(SAMPLE_PROMPTS):
        target_col = col1 if i % 2 == 0 else col2
        if target_col.button(f"{icon}  {prompt_text}", key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_prompt = prompt_text
            st.rerun()

elif st.session_state.messages:
    # ── Active Chat Header ─────────────────────────────────────────────────
    st.markdown(
        """
        <div class="chat-header">
          <span class="chat-header-title">Zenic</span>
          <div class="chat-header-status">
            <div class="chat-header-dot"></div>
            <span>ENGINE ACTIVE</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Metrics are stored on the message rather than rendered only in the run
        # that produced them, so they survive every later rerun (profile update,
        # PDF download, any widget interaction) instead of vanishing.
        if msg.get("metrics"):
            render_metric_cards(msg["metrics"])
            st.markdown("<br>", unsafe_allow_html=True)
        st.write(msg["content"])

typed_prompt = st.chat_input("Ask Zenic — nutrition, training, or a precision plan…", max_chars=4000)

prompt = typed_prompt or st.session_state.pending_prompt
if st.session_state.pending_prompt:
    st.session_state.pending_prompt = None

#: Shown when a turn fails. Deliberately free of internals — the detail goes to
#: the logs under the turn's correlation id, not to the user's screen.
_GENERIC_ERROR = (
    "Something went wrong while processing that. The issue has been logged — "
    "please try again in a moment."
)


#: What each graph node is doing, in the user's terms. A retrieval turn is
#: dominated by cross-encoder reranking on CPU (~30s of a ~35s turn), so a single
#: opaque spinner leaves the user with no idea whether anything is happening.
_NODE_LABELS = {
    "safety_check": "Checking your message",
    "router": "Understanding your question",
    "profile_check": "Reviewing your profile",
    "profile_gather": "Working out what's missing",
    "rag_retrieval": "Searching the knowledge base",
    "calculator": "Running your numbers",
    "food_retrieval": "Finding foods that fit",
    "exercise_retrieval": "Selecting exercises",
    "plan_compose": "Composing your plan",
    "pdf_generate": "Building your PDF",
    "data_ingestion": "Loading your week",
    "trend_analysis": "Analysing trends",
    "insight_generation": "Drawing out insights",
    "generate": "Writing the answer",
    "safety_response": "Preparing a response",
}

#: The stage that follows routing, per intent. LangGraph reports a node when it
#: *finishes*, so naming the node that just completed would leave a stale label
#: on screen for the whole of the next (and by far the longest) step. Announcing
#: the upcoming stage instead keeps the label truthful.
_STAGE_AFTER_ROUTER = {
    "nutrition_qa": "Searching the knowledge base",
    "calculate": "Reviewing your profile",
    "meal_plan": "Reviewing your profile",
    "workout_plan": "Reviewing your profile",
    "weekly_summary": "Loading your week",
    "general_chat": "Writing the answer",
}

_MERGED_DICT_FIELDS = ("tool_results", "plan_data", "user_profile")


def _next_stage_label(completed_node: str, state: dict) -> str:
    """Human label for the work that starts now that ``completed_node`` is done."""
    if completed_node == "router":
        return _STAGE_AFTER_ROUTER.get(state.get("intent", ""), "Working on it")
    if completed_node in ("rag_retrieval", "calculator", "plan_compose"):
        return _NODE_LABELS["generate"]
    if completed_node in ("food_retrieval", "exercise_retrieval"):
        return _NODE_LABELS["plan_compose"]
    return _NODE_LABELS.get(completed_node, completed_node.replace("_", " ").title())


def _merge_partial(state: dict, partial) -> None:
    """Fold one node's returned fields into the accumulated state."""
    if not isinstance(partial, dict):
        return
    for key, value in partial.items():
        if key == "messages":
            new = value if isinstance(value, list) else [value]
            state["messages"] = list(state.get("messages") or []) + new
        elif key in _MERGED_DICT_FIELDS and isinstance(value, dict):
            state[key] = {**(state.get(key) or {}), **value}
        else:
            state[key] = value


def run_turn(messages: list[dict], user_profile: dict, on_stage=None) -> tuple[dict | None, str | None]:
    """Invoke the agent, reporting each stage as it starts.

    Streams the graph rather than calling invoke() so the UI can name the stage
    currently running. Returns (final_state, error_message).
    """
    correlation_id = bind_correlation_id()
    state = initial_state(
        [{"role": m["role"], "content": m["content"]} for m in messages[-12:]],
        user_profile,
    )
    try:
        accumulated = dict(state)
        for step in zenic_app.stream(state):
            for node_name, partial in step.items():
                _merge_partial(accumulated, partial)
                if on_stage:
                    on_stage(_next_stage_label(node_name, accumulated))
        return accumulated, None
    except ZenicError as exc:
        # Operational failures carry a message written for the user.
        logger.warning("turn failed", extra={"error_type": type(exc).__name__})
        return None, str(exc)
    except Exception:
        logger.error("unhandled error during turn")
        return None, f"{_GENERIC_ERROR} (reference: {correlation_id})"


def extract_reply(final_state: dict) -> str:
    """Last assistant message from the final state, whatever shape it is in."""
    for message in reversed(final_state.get("messages") or []):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "type", None)
        if role in ("ai", "assistant"):
            return (
                message.get("content", "")
                if isinstance(message, dict)
                else getattr(message, "content", "")
            )
    return "Sorry, I couldn't generate a response."


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        status = st.status("Reading your message", expanded=False)
        seen_stages: list[str] = []

        def report(stage: str) -> None:
            # Repeated labels (e.g. two retrieval nodes in a plan) would otherwise
            # log the same line twice.
            if not seen_stages or seen_stages[-1] != stage:
                seen_stages.append(stage)
                status.write(stage)
            status.update(label=stage)

        final_state, error = run_turn(
            st.session_state.messages, st.session_state.user_profile, on_stage=report
        )
        status.update(
            label="Couldn't complete that" if error else "Done",
            state="error" if error else "complete",
        )

        metrics = None
        if error:
            st.error(error)
            reply = error
        else:
            reply = extract_reply(final_state)
            if final_state.get("intent") == "calculate":
                metrics = final_state.get("tool_results") or {}
                render_metric_cards(metrics)
                st.markdown("<br>", unsafe_allow_html=True)
            st.write(reply)

    assistant_message = {"role": "assistant", "content": reply}
    if metrics:
        assistant_message["metrics"] = metrics
    st.session_state.messages.append(assistant_message)
    st.session_state.messages = st.session_state.messages[-40:]

    if final_state:
        # The sidebar renders near the top of the script, before this turn ran, so
        # a profile or PDF produced here is only visible after a rerun — without
        # one the sidebar shows stale values for a whole extra turn.
        needs_rerun = False

        profile = final_state.get("user_profile")
        if profile and profile != st.session_state.user_profile:
            st.session_state.user_profile = profile
            needs_rerun = True

        pdf_path = (final_state.get("tool_results") or {}).get("pdf_path")
        if pdf_path and pdf_path != st.session_state.pdf_path:
            if st.session_state.pdf_path:
                old_path = Path(st.session_state.pdf_path)
                old_path.unlink(missing_ok=True)
                old_path.parent.rmdir()
            st.session_state.pdf_path = pdf_path
            needs_rerun = True

        if needs_rerun:
            st.rerun()
