"""Unexpected graph failures must not write prompt-bearing exception text to logs."""
import logging

from zenic.agent import turn


def test_unhandled_graph_exception_logs_type_only(monkeypatch, caplog):
    class BrokenGraph:
        def stream(self, _state):
            raise RuntimeError("private health question in provider response")

    monkeypatch.setattr(turn, "app", BrokenGraph())
    with caplog.at_level(logging.ERROR):
        final, error, _timings = turn.run_turn(
            [{"role": "user", "content": "private health question"}], {}
        )
    assert final is None
    assert "Reference:" in error
    assert any(getattr(record, "error_type", None) == "RuntimeError" for record in caplog.records)
    assert "private health question" not in caplog.text
