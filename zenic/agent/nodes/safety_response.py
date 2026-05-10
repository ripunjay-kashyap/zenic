"""Return a safe refusal message when the safety flag is set."""
from zenic.agent.state import ZenicState


def run(state: ZenicState) -> dict:
    reason = state.get("safety_reason", "")
    intro = (
        f"I can't help with that — your request involves {reason}."
        if reason
        else "I'm not able to help with that request."
    )
    message = (
        f"{intro} "
        "Zenic is designed to support healthy nutrition and fitness goals. "
        "Please consult a qualified healthcare professional for medical advice."
    )
    return {"messages": [{"role": "assistant", "content": message}]}
