"""Central Identity and Persona Configuration for ADAM.
Ensures single source of truth for assistant identity, creator attribution,
and deterministic identity query responses.
"""

from typing import Optional

ASSISTANT_NAME = "ADAM"
ASSISTANT_FULL_NAME = "Advanced Desktop AI Mate"
ASSISTANT_DESCRIPTION = "Advanced Desktop AI Mate"
ASSISTANT_ROLE = "Virtual assistant"
ASSISTANT_CREATOR = "Sujeeth Kani"
ASSISTANT_VERSION = "3.1.0"


def get_assistant_name() -> str:
    """Return the canonical assistant name."""
    return ASSISTANT_NAME


def get_assistant_identity_response() -> str:
    """Deterministic response to 'What is your name?' or 'Who are you?'."""
    return f"My name is {ASSISTANT_NAME}. I am your {ASSISTANT_ROLE.lower()}."


def get_assistant_short_identity() -> str:
    """Short casual self-identification."""
    return f"I'm {ASSISTANT_NAME}, your {ASSISTANT_ROLE.lower()}."


def get_creator_response() -> str:
    """Deterministic response to 'Who created you?'."""
    return f"I was created by {ASSISTANT_CREATOR} as {ASSISTANT_NAME} ({ASSISTANT_FULL_NAME})."


def get_user_name_response(user_name: Optional[str]) -> str:
    """Deterministic response to 'What is my name?' based on stored personal memory."""
    if user_name and str(user_name).strip():
        clean_name = str(user_name).strip().title()
        return f"Your name is {clean_name}."
    return "I don't know your name yet. What should I call you?"
