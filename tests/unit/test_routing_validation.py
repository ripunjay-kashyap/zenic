"""The operator's live routing check must fail for every route mismatch."""

import pytest

from scripts import rag_vs_api_check


@pytest.mark.parametrize(
    "should_use_rag,nodes,expected_exit",
    [
        (True, ["rag_retrieval", "usda_api"], 1),
        (False, ["rag_retrieval"], 1),
        (True, ["generate"], 1),
        (True, ["rag_retrieval"], 0),
        (False, ["rag_retrieval", "usda_api"], 0),
    ],
)
def test_routing_check_exit_status(monkeypatch, should_use_rag, nodes, expected_exit):
    monkeypatch.setattr(
        rag_vs_api_check, "_CASES",
        [{"query": "synthetic routing probe", "should_use_rag": should_use_rag}],
    )
    monkeypatch.setattr(
        rag_vs_api_check, "run_with_trace", lambda _: {"tools_called": nodes},
    )
    assert rag_vs_api_check.main() == expected_exit


def test_missing_food_can_abstain_when_api_has_no_relevant_evidence(monkeypatch):
    monkeypatch.setattr(
        rag_vs_api_check, "_CASES", [{"query": "unknown food", "should_use_rag": False}],
    )
    monkeypatch.setattr(rag_vs_api_check, "run_with_trace", lambda _: {
        "tools_called": ["rag_retrieval", "generate"],
        "final_state": {"messages": [{
            "role": "assistant", "content": rag_vs_api_check.NO_EVIDENCE_RESPONSE,
        }]},
    })
    assert rag_vs_api_check.main() == 0
