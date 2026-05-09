import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from zenic.agent.graph import app as zenic_app
from zenic.agent.state import ZenicState

# ---------------------------------------------------------------------------
# UI Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Zenic", page_icon="🧬", layout="centered")

def load_css(file_name):
    with open(file_name) as f:
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
    st.markdown(
        "<p style='font-family:Space Mono,monospace;font-size:0.62rem;"
        "letter-spacing:0.22em;text-transform:uppercase;color:#22c55e;"
        "margin-bottom:1.25rem;'>// Bio-Data</p>",
        unsafe_allow_html=True,
    )

    profile = st.session_state.user_profile
    REQUIRED = ["weight_kg", "height_cm", "age", "gender", "activity_level", "goal"]
    completed = [f for f in REQUIRED if profile.get(f)]
    percent = int((len(completed) / len(REQUIRED)) * 100)

    st.markdown(
        f"""
        <div style='background:rgba(34,197,94,0.05);border:1px solid rgba(34,197,94,0.18);
        border-radius:10px;padding:12px 14px;margin-bottom:1.5rem;'>
            <div style='display:flex;justify-content:space-between;margin-bottom:9px;'>
                <span style='font-family:Space Mono,monospace;font-size:0.58rem;
                letter-spacing:0.14em;text-transform:uppercase;color:rgba(221,225,244,0.5);'>
                PROFILE SYNC</span>
                <span style='font-family:Space Mono,monospace;font-size:0.65rem;
                font-weight:700;color:#22c55e;'>{percent}%</span>
            </div>
            <div style='width:100%;height:4px;background:rgba(255,255,255,0.05);border-radius:2px;'>
                <div style='width:{percent}%;height:100%;background:linear-gradient(90deg,#22c55e,#a78bfa);
                border-radius:2px;box-shadow:0 0 10px rgba(34,197,94,0.4);
                transition:width 0.5s ease;'></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if profile:
        field_icons = {
            "age": "🎂", "weight_kg": "⚖️", "height_cm": "📏", "gender": "👤",
            "activity_level": "🏃", "goal": "🎯", "dietary_restrictions": "🥗",
            "experience_level": "💪", "available_days": "📅", "equipment": "🏋️",
        }
        labels = {
            "age": "Age", "weight_kg": "Weight", "height_cm": "Height", "gender": "Gender",
            "activity_level": "Activity", "goal": "Goal", "dietary_restrictions": "Diet",
            "experience_level": "Experience", "available_days": "Schedule", "equipment": "Gym",
        }
        for key, icon in field_icons.items():
            val = profile.get(key)
            if val is not None and val != "":
                display_val = format_profile_value(key, val)
                st.markdown(
                    f"""<div style='display:flex;align-items:center;gap:10px;padding:7px 0;
                    border-bottom:1px solid rgba(255,255,255,0.04);'>
                        <span style='font-size:1rem;line-height:1;'>{icon}</span>
                        <div>
                            <div style='font-family:Space Mono,monospace;font-size:0.5rem;
                            letter-spacing:0.14em;text-transform:uppercase;
                            color:rgba(232,237,255,0.34);margin-bottom:1px;'>
                            {labels.get(key, key.title())}</div>
                            <div style='font-family:Outfit,sans-serif;font-size:0.84rem;
                            color:#e8edff;font-weight:400;'>{display_val}</div>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
    else:
        st.caption("Commence chat to build biometric profile.")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
        st.markdown(
            "<p style='font-family:Space Mono,monospace;font-size:0.6rem;"
            "letter-spacing:0.18em;text-transform:uppercase;color:#22c55e;"
            "margin-bottom:0.75rem;'>// Vault</p>",
            unsafe_allow_html=True,
        )
        with open(st.session_state.pdf_path, "rb") as f:
            st.download_button(
                "DOWNLOAD PLAN",
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
if not st.session_state.messages:
    # ── Hero Section ──────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-wrap">

          <!-- Bio-glyph: dual rotating hexagons + nucleus -->
          <div class="hero-glyph-ring">
            <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
              <!-- Outer ring - spins CW -->
              <g class="glyph-outer">
                <polygon points="32,4 56,18 56,46 32,60 8,46 8,18"
                         stroke="#22c55e" stroke-width="0.8" fill="none" opacity="0.55"/>
                <circle cx="32" cy="4" r="2.5" fill="#22c55e" opacity="0.85"/>
              </g>
              <!-- Inner ring - spins CCW -->
              <g class="glyph-inner">
                <polygon points="32,14 48,23 48,41 32,50 16,41 16,23"
                         stroke="#a78bfa" stroke-width="0.6" fill="none" opacity="0.45"/>
                <circle cx="48" cy="23" r="1.8" fill="#a78bfa" opacity="0.75"/>
              </g>
              <!-- Nucleus -->
              <circle cx="32" cy="32" r="5" fill="rgba(34,197,94,0.12)"
                      stroke="#22c55e" stroke-width="0.8"/>
              <circle cx="32" cy="32" r="2.5" fill="#22c55e" opacity="0.9"/>
            </svg>
          </div>

          <!-- Wordmark -->
          <span class="hero-title">ZENIC</span>

          <!-- Tagline row -->
          <div class="hero-tagline-row">
            <div class="hero-tagline-line"></div>
            <div class="hero-badge">
              <div class="hero-dot"></div>
              BIO-INTELLIGENCE ENGINE
            </div>
            <div class="hero-tagline-line right"></div>
          </div>

          <p class="hero-sub">
            Precision nutrition synthesis &amp; biomolecular analysis — built for the serious athlete.
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

else:
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

typed_prompt = st.chat_input("Feed the biometric intelligence...")

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
