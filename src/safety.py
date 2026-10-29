"""Safety checks for crisis-related language."""

from __future__ import annotations

from dataclasses import dataclass


CRISIS_PHRASES: tuple[str, ...] = (
    "kill myself",
    "suicide",
    "end my life",
    "end it all",
    "want to die",
    "do not want to live",
    "dont want to live",
    "give up",
    "cant go on",
    "cannot go on",
    "better off dead",
    "hurt myself",
    "self harm",
    "selfharm",
)


@dataclass(frozen=True)
class CrisisAssessment:
    """Structured safety result for urgent language."""

    flagged: bool
    matched_phrases: list[str]
    message: str


def detect_crisis_language(text: str) -> CrisisAssessment:
    """Detect urgent self-harm or crisis language in free text.

    The goal is not to diagnose, but to enforce a conservative escalation path
    for obviously high-risk phrases that should never be treated as normal text.
    """
    normalized = " ".join(str(text or "").lower().split())
    matched: list[str] = []

    for phrase in CRISIS_PHRASES:
        if phrase in normalized:
            matched.append(phrase)

    if matched:
        return CrisisAssessment(
            flagged=True,
            matched_phrases=sorted(set(matched)),
            message=(
                "Urgent self-harm or crisis language detected. "
                "This input should be treated as high concern and escalated for immediate support."
            ),
        )

    return CrisisAssessment(flagged=False, matched_phrases=[], message="")
