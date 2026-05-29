import os
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from zenic.agent.graph import app as zenic_app
from zenic.agent.state import ZenicState


def _greeting() -> str:
    """Time-of-day greeting for the hero."""
    hour = datetime.now().hour
    if hour < 5:   return "Still up"
    if hour < 12:  return "Good morning"
    if hour < 17:  return "Good afternoon"
    if hour < 22:  return "Good evening"
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
    st.markdown(
        f"""
        <div class="sync-card">
            <div class="sync-row">
                <span class="sync-label">Bio-Sync</span>
                <span class="sync-pct">{percent}%</span>
            </div>
            <div class="sync-track">
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
                display_val = format_profile_value(key, val)
                rows_html.append(
                    f"""<div class="field-row">
                        <span class="field-icon">{icon}</span>
                        <div class="field-body">
                            <span class="field-label">{labels.get(key, key.title())}</span>
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
        st.write(msg["content"])

typed_prompt = st.chat_input("Ask Zenic — nutrition, training, or a precision plan…")

prompt = typed_prompt or st.session_state.pending_prompt
if st.session_state.pending_prompt:
    st.session_state.pending_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    initial_state: ZenicState = {
        "messages": [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
        "user_profile": st.session_state.user_profile,
        "intent": "",
        "profile_complete": False,
        "missing_fields": [],
        "awaiting_input": False,
        "retrieved_context": [],
        "tool_results": {},
        "plan_data": {},
        "safety_flag": False,
        "safety_reason": "",
    }

    with st.chat_message("assistant"):
        with st.spinner("Processing bio-data..."):
            final_state = zenic_app.invoke(initial_state)

        assistant_msgs = [m for m in final_state.get("messages", []) if getattr(m, "type", None) == "ai"]
        reply = assistant_msgs[-1].content if assistant_msgs else "Sorry, I couldn't generate a response."

        if final_state.get("intent") == "calculate":
            render_metric_cards(final_state.get("tool_results"))
            st.markdown("<br>", unsafe_allow_html=True)

        st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

    if final_state.get("user_profile"):
        st.session_state.user_profile = final_state["user_profile"]

    pdf_path = (final_state.get("tool_results") or {}).get("pdf_path")
    if pdf_path:
        st.session_state.pdf_path = pdf_path
        st.rerun()
