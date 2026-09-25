"""Node behaviour: degradation on failure, routing inputs, and data handling."""
import pytest

from zenic.agent.graph import (
    _route_after_calculator,
    _route_after_profile_check,
    _route_after_router,
    _route_after_safety,
    initial_state,
)
from zenic.agent.nodes import (
    calculator,
    data_ingestion,
    exercise_retrieval,
    generate,
    insight_generation,
    pdf_generate,
    profile_check,
    profile_gather,
    rag_retrieval,
    router,
    trend_analysis,
)
from zenic.errors import LLMError, RetrievalError, ZenicError

_COMPLETE_PROFILE = {
    "weight_kg": 80,
    "height_cm": 178,
    "age": 28,
    "gender": "male",
    "activity_level": "moderate",
    "goal": "cutting",
}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_safety_flag_diverts_to_the_refusal_node():
    assert _route_after_safety({"safety_flag": True}) == "safety_response"
    assert _route_after_safety({"safety_flag": False}) == "router"


@pytest.mark.parametrize(
    "intent,expected",
    [
        ("nutrition_qa", "rag_retrieval"),
        ("calculate", "profile_check"),
        ("meal_plan", "profile_check"),
        ("workout_plan", "profile_check"),
        ("weekly_summary", "data_ingestion"),
        ("general_chat", "generate"),
        ("", "generate"),
    ],
)
def test_router_destinations(intent, expected):
    assert _route_after_router({"intent": intent}) == expected


def test_incomplete_profile_always_asks_first():
    state = {"intent": "calculate", "profile_complete": False}
    assert _route_after_profile_check(state) == "profile_gather"


def test_meal_plan_runs_the_calculator_before_retrieval():
    """Without this, plan_compose is prompted with empty macro targets."""
    state = {"intent": "meal_plan", "profile_complete": True}
    assert _route_after_profile_check(state) == "calculator"
    assert _route_after_calculator(state) == "food_retrieval"


def test_calculate_goes_straight_to_generation():
    assert _route_after_calculator({"intent": "calculate"}) == "generate"


# ---------------------------------------------------------------------------
# Router node
# ---------------------------------------------------------------------------

def test_router_defaults_when_the_llm_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        router, "chat_completion_json", lambda *a, **k: (_ for _ in ()).throw(LLMError("down"))
    )
    assert router.run({"messages": []})["intent"] == router.DEFAULT_INTENT


def test_router_rejects_an_intent_outside_the_enum(monkeypatch):
    """An unknown label would otherwise fall through every routing branch."""
    monkeypatch.setattr(router, "chat_completion_json", lambda *a, **k: {"intent": "buy_socks"})
    assert router.run({"messages": []})["intent"] == router.DEFAULT_INTENT


def test_router_accepts_a_valid_intent(monkeypatch):
    monkeypatch.setattr(router, "chat_completion_json", lambda *a, **k: {"intent": "meal_plan"})
    assert router.run({"messages": []})["intent"] == "meal_plan"


def test_pending_profile_reply_keeps_workflow_when_router_fails(monkeypatch):
    monkeypatch.setattr(
        router, "chat_completion_json", lambda *a, **k: (_ for _ in ()).throw(LLMError("down"))
    )
    state = initial_state(
        [{"role": "user", "content": "male and maintenance"}],
        {"age": 28},
        pending_intent="calculate",
        pending_missing_fields=["gender", "goal"],
    )
    assert state["awaiting_input"] is True
    assert state["missing_fields"] == ["gender", "goal"]
    assert router.run(state) == {"intent": "calculate", "awaiting_input": True}


def test_pending_profile_reply_can_start_a_new_request(monkeypatch):
    def classify(messages, **kwargs):
        assert "gender, goal" in messages[1]["content"]
        return {"intent": "nutrition_qa"}

    monkeypatch.setattr(router, "chat_completion_json", classify)
    state = initial_state(
        [{"role": "user", "content": "Actually, tell me about vitamin D"}],
        pending_intent="calculate",
        pending_missing_fields=["gender", "goal"],
    )
    assert router.run(state) == {"intent": "nutrition_qa", "awaiting_input": False}


def test_completed_profile_clears_pending_request(monkeypatch):
    monkeypatch.setattr(
        profile_check,
        "chat_completion_json",
        lambda *a, **k: {"gender": "male", "goal": "maintenance"},
    )
    state = initial_state(
        [{"role": "user", "content": "male and maintenance"}],
        {"age": 28, "weight_kg": 75, "height_cm": 178, "activity_level": "moderate"},
        pending_intent="calculate",
        pending_missing_fields=["gender", "goal"],
    )
    result = profile_check.run(state)
    assert result["profile_complete"] is True
    assert result["awaiting_input"] is False
    assert result["missing_fields"] == []


# ---------------------------------------------------------------------------
# profile_check
# ---------------------------------------------------------------------------

def test_profile_check_does_not_mutate_the_incoming_profile(monkeypatch):
    monkeypatch.setattr(
        profile_check, "chat_completion_json", lambda *a, **k: {"weight_kg": 90}
    )
    original = {"weight_kg": 80}
    state = {
        "intent": "calculate",
        "user_profile": original,
        "messages": [{"role": "user", "content": "I now weigh 90kg"}],
    }
    result = profile_check.run(state)
    assert result["user_profile"]["weight_kg"] == 90
    assert original == {"weight_kg": 80}


def test_profile_check_normalises_extracted_enums(monkeypatch):
    monkeypatch.setattr(
        profile_check,
        "chat_completion_json",
        lambda *a, **k: {"activity_level": "moderately active", "goal": "fat loss"},
    )
    state = {
        "intent": "calculate",
        "user_profile": {},
        "messages": [{"role": "user", "content": "I'm moderately active, cutting"}],
    }
    profile = profile_check.run(state)["user_profile"]
    assert profile["activity_level"] == "moderate"
    assert profile["goal"] == "cutting"


def test_profile_check_survives_a_failed_extraction(monkeypatch):
    monkeypatch.setattr(
        profile_check,
        "chat_completion_json",
        lambda *a, **k: (_ for _ in ()).throw(LLMError("down")),
    )
    state = {
        "intent": "calculate",
        "user_profile": dict(_COMPLETE_PROFILE),
        "messages": [{"role": "user", "content": "tdee?"}],
    }
    result = profile_check.run(state)
    assert result["profile_complete"] is True


def test_profile_check_makes_no_llm_call_without_required_fields():
    def _fail(*a, **k):
        pytest.fail("weekly_summary needs no profile extraction")

    state = {"intent": "weekly_summary", "user_profile": {}, "messages": []}
    assert profile_check.run(state)["profile_complete"] is True


# ---------------------------------------------------------------------------
# profile_gather
# ---------------------------------------------------------------------------

def test_profile_gather_falls_back_to_a_static_prompt(monkeypatch):
    monkeypatch.setattr(
        profile_gather, "chat_completion", lambda *a, **k: (_ for _ in ()).throw(LLMError("down"))
    )
    result = profile_gather.run({"missing_fields": ["age", "weight_kg"], "intent": "calculate"})
    content = result["messages"][0]["content"]
    assert "your age" in content
    assert "your weight in kg" in content
    assert result["awaiting_input"] is True


# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------

def test_calculator_produces_every_metric():
    results = calculator.run({"user_profile": dict(_COMPLETE_PROFILE)})["tool_results"]
    assert set(results) >= {
        "bmr", "tdee", "protein_g", "carbs_g", "fat_g", "protein_min_g", "protein_max_g"
    }


def test_calculator_names_the_missing_field():
    """Previously a bare KeyError with no indication of what was absent."""
    with pytest.raises(ZenicError, match="activity_level"):
        calculator.run({"user_profile": {k: v for k, v in _COMPLETE_PROFILE.items()
                                         if k != "activity_level"}})


def test_calculator_rejects_an_impossible_energy_estimate():
    profile = dict(_COMPLETE_PROFILE)
    profile.update(weight_kg=20, height_cm=50, age=120, gender="female")
    with pytest.raises(ZenicError, match="valid energy estimate"):
        calculator.run({"user_profile": profile})


def test_calculator_preserves_earlier_tool_results():
    state = {"user_profile": dict(_COMPLETE_PROFILE), "tool_results": {"prior": 1}}
    assert calculator.run(state)["tool_results"]["prior"] == 1


# ---------------------------------------------------------------------------
# exercise_retrieval
# ---------------------------------------------------------------------------

def test_cutting_goal_selects_the_conditioning_split():
    """This branch tested for "fat_loss", which the normaliser never produces."""
    assert exercise_retrieval._select_split(4, "cutting") == "full_body_cardio"


@pytest.mark.parametrize("days,split", [(3, "full_body"), (4, "upper_lower"), (5, "ulppl"), (6, "ppl")])
def test_split_follows_available_days(days, split):
    assert exercise_retrieval._select_split(days, "bulking") == split


def test_more_than_six_days_caps_at_ppl():
    assert exercise_retrieval._select_split(9, "bulking") == "ppl"


def test_non_numeric_days_fall_back_to_full_body():
    assert exercise_retrieval._select_split("lots", "bulking") == "full_body"


def test_exercise_retrieval_degrades_when_the_index_is_down(monkeypatch):
    monkeypatch.setattr(
        exercise_retrieval, "retrieve", lambda q: (_ for _ in ()).throw(RetrievalError("down"))
    )
    result = exercise_retrieval.run({"user_profile": {"goal": "bulking", "available_days": 4}})
    assert result["tool_results"]["exercise_chunks"] == []
    assert result["tool_results"]["split_type"] == "upper_lower"


# ---------------------------------------------------------------------------
# rag_retrieval
# ---------------------------------------------------------------------------

def test_rag_retrieval_returns_empty_when_the_index_is_down(monkeypatch):
    monkeypatch.setattr(
        rag_retrieval, "retrieve", lambda q: (_ for _ in ()).throw(RetrievalError("down"))
    )
    monkeypatch.setattr(rag_retrieval, "get_settings", lambda: type("S", (), {"usda_api_key": None})())
    state = {"messages": [{"role": "user", "content": "protein in chicken"}]}
    assert rag_retrieval.run(state)["retrieved_context"] == []


def test_good_retrieval_does_not_call_the_live_api(monkeypatch):
    monkeypatch.setattr(
        rag_retrieval, "retrieve", lambda q: [{"text": "t", "metadata": {}, "rerank_score": 5.0}]
    )
    monkeypatch.setattr(
        rag_retrieval, "_usda_fallback", lambda q: pytest.fail("RAG-first rule violated")
    )
    state = {"messages": [{"role": "user", "content": "protein in chicken"}]}
    monkeypatch.setattr(rag_retrieval, "rerank", lambda q, chunks: chunks)
    result = rag_retrieval.run(state)
    assert "api_fallback_used" not in (result.get("tool_results") or {})


def test_referential_follow_up_is_rewritten_before_retrieval(monkeypatch):
    previous = "According to NIH, what is vitamin D intake for ages 19 to 70?"
    latest = "What does that same fact sheet say for people older than 70?"
    standalone = "What is NIH vitamin D intake for adults older than 70?"
    seen = []
    monkeypatch.setattr(rag_retrieval, "chat_completion", lambda *a, **k: standalone)
    monkeypatch.setattr(rag_retrieval, "retrieve", lambda query: seen.append(query) or [
        {"text": "20 mcg (800 IU)", "metadata": {"source": "NIH_ODS"}, "rerank_score": 5.0}
    ])
    state = {"messages": [
        {"role": "user", "content": previous},
        {"role": "assistant", "content": "15 mcg (600 IU)"},
        {"role": "user", "content": latest},
    ]}
    result = rag_retrieval.run(state)
    assert seen == [standalone]
    assert result["retrieval_query"] == standalone


def test_poor_retrieval_falls_back_to_the_live_api(monkeypatch):
    monkeypatch.setattr(
        rag_retrieval, "retrieve", lambda q: [{"text": "t", "metadata": {}, "rerank_score": -2.0}]
    )
    monkeypatch.setattr(
        rag_retrieval, "get_settings", lambda: type("S", (), {"usda_api_key": "key"})()
    )
    monkeypatch.setattr(
        rag_retrieval, "_usda_fallback", lambda q: [{"text": "live", "metadata": {}, "rerank_score": 5}]
    )
    state = {"messages": [{"role": "user", "content": "protein in quinoa"}]}
    monkeypatch.setattr(rag_retrieval, "rerank", lambda q, chunks: chunks)
    result = rag_retrieval.run(state)
    assert result["tool_results"]["api_fallback_used"] == "usda_api"


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def test_generate_hides_internal_bookkeeping_from_the_prompt(monkeypatch):
    captured = {}

    def _fake(query, context, intent="", history=None):
        captured["context"] = context
        return "answer"

    monkeypatch.setattr(generate, "rag_generate", _fake)
    state = {
        "messages": [{"role": "user", "content": "q"}],
        "tool_results": {
            "tdee": 2400,
            "api_fallback_used": "usda_api",
            "pdf_path": "/tmp/x.pdf",
            "food_chunks": [{"text": "a" * 5000}],
        },
    }
    generate.run(state)
    calc_text = captured["context"][0]["text"]
    assert "tdee: 2400" in calc_text
    assert "api_fallback_used" not in calc_text
    assert "pdf_path" not in calc_text
    assert "food_chunks" not in calc_text


def test_generate_does_not_mutate_retrieved_context(monkeypatch):
    monkeypatch.setattr(generate, "rag_generate", lambda *a, **k: "answer")
    context = [{"text": "chunk", "metadata": {}}]
    state = {
        "messages": [{"role": "user", "content": "q"}],
        "retrieved_context": context,
        "tool_results": {"tdee": 2400},
    }
    generate.run(state)
    assert len(context) == 1


# ---------------------------------------------------------------------------
# weekly summary chain
# ---------------------------------------------------------------------------

def test_stale_demo_data_still_produces_a_week(monkeypatch):
    """The bundled dataset is fixed-date; a strict window returns nothing."""
    entries = [
        {"date": f"2020-01-0{i}", "calories": 2000 + i, "protein_g": 150, "workout_planned": True}
        for i in range(1, 9)
    ]
    monkeypatch.setattr(data_ingestion, "_load_entries", lambda: entries)
    weekly = data_ingestion.run({})["tool_results"]["weekly_data"]
    assert len(weekly) == 7
    assert weekly[-1]["date"] == "2020-01-08"


def test_missing_data_file_yields_a_placeholder_week(monkeypatch):
    monkeypatch.setattr(data_ingestion, "_load_entries", lambda: [])
    weekly = data_ingestion.run({})["tool_results"]["weekly_data"]
    assert len(weekly) == 7
    assert all(day["calories"] == 0 for day in weekly)


def test_trend_analysis_sorts_by_date_for_weight_change():
    data = [
        {"date": "2026-01-03", "calories": 2000, "protein_g": 150, "weight_kg": 81.0},
        {"date": "2026-01-01", "calories": 2000, "protein_g": 150, "weight_kg": 82.0},
    ]
    stats = trend_analysis.run({"tool_results": {"weekly_data": data}})["tool_results"]["weekly_stats"]
    assert stats["weight_change_kg"] == pytest.approx(-1.0)


def test_trend_analysis_skips_non_numeric_entries():
    data = [
        {"date": "2026-01-01", "calories": 2000, "protein_g": 150},
        {"date": "2026-01-02", "calories": "n/a", "protein_g": None},
    ]
    stats = trend_analysis.run({"tool_results": {"weekly_data": data}})["tool_results"]["weekly_stats"]
    assert stats["avg_calories"] == pytest.approx(2000)


def test_an_untracked_week_reports_no_stats():
    data = [{"date": "2026-01-01", "calories": 0, "protein_g": 0, "workout_planned": False}]
    stats = trend_analysis.run({"tool_results": {"weekly_data": data}})["tool_results"]["weekly_stats"]
    assert stats == {}


def test_insights_explain_the_absence_of_data_without_calling_the_llm(monkeypatch):
    monkeypatch.setattr(
        insight_generation, "chat_completion", lambda *a, **k: pytest.fail("no LLM call expected")
    )
    plan = insight_generation.run({"tool_results": {"weekly_stats": {}}})["plan_data"]
    assert "No tracking data" in plan["insights"]


def test_insights_degrade_when_the_llm_fails(monkeypatch):
    monkeypatch.setattr(
        insight_generation,
        "chat_completion",
        lambda *a, **k: (_ for _ in ()).throw(LLMError("down")),
    )
    plan = insight_generation.run({"tool_results": {"weekly_stats": {"avg_calories": 2100}}})
    assert plan["plan_data"]["weekly_stats"] == {"avg_calories": 2100}


# ---------------------------------------------------------------------------
# pdf_generate
# ---------------------------------------------------------------------------

def test_each_pdf_gets_its_own_path():
    """A fixed filename let concurrent users overwrite each other's plan."""
    assert pdf_generate._output_path("meal_plan") != pdf_generate._output_path("meal_plan")


@pytest.mark.parametrize("intent", ["meal_plan", "workout_plan", "weekly_summary", "general_chat"])
def test_pdf_renders_for_every_intent(intent):
    state = {"intent": intent, "plan_data": {"days": [], "notes": "n", "insights": "i"}}
    result = pdf_generate.run(state)
    path = result["tool_results"]["pdf_path"]
    assert path.endswith(".pdf")
    with open(path, "rb") as f:
        assert f.read(4) == b"%PDF"


def test_weekly_summary_renders_multi_line_insights():
    """Real insights are several lines. multi_cell(0, ...) leaves the cursor at
    the cell's right edge, so a second call got zero width and raised
    'Not enough horizontal space to render a single character'."""
    state = {
        "intent": "weekly_summary",
        "plan_data": {
            "weekly_stats": {"avg_calories": 2280.5, "avg_protein_g": 178.0},
            "insights": (
                "- Your average intake was 2,280 kcal against a 2,400 target.\n"
                "- Protein was consistent at 178 g/day.\n"
                "1. Workout adherence hit 80% this week.\n"
                "Consider adding one more session next week."
            ),
        },
    }
    path = pdf_generate.run(state)["tool_results"]["pdf_path"]
    with open(path, "rb") as f:
        assert f.read(4) == b"%PDF"


def test_generic_template_renders_multi_line_values():
    """The generic fallback chains multi_cell the same way."""
    state = {"intent": "", "plan_data": {"notes": ["first line", "second line", "third line"]}}
    path = pdf_generate.run(state)["tool_results"]["pdf_path"]
    with open(path, "rb") as f:
        assert f.read(4) == b"%PDF"


def test_pdf_survives_a_malformed_plan():
    """plan_data comes from an LLM, so its shape is not guaranteed."""
    state = {"intent": "workout_plan", "plan_data": {"days": [{"exercises": ["Squat", "Bench"]}]}}
    assert pdf_generate.run(state)["tool_results"]["pdf_path"].endswith(".pdf")


# ---------------------------------------------------------------------------
# initial_state
# ---------------------------------------------------------------------------

def test_initial_state_populates_every_field():
    from zenic.agent.state import ZenicState

    state = initial_state()
    assert set(state) == set(ZenicState.__annotations__)
