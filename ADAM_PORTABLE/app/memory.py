"""Persistent Personal Memory Subsystem for ADAM.
Handles explicit user facts, preferences, identity resolution,
and safeguards against accidental over-generalization or secret storage.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from app.identity import (
    get_creator_response,
    get_assistant_identity_response,
    get_assistant_name,
    get_user_name_response,
)
from app.storage import LocalStore

logger = logging.getLogger("ADAM.Memory")

# Block storage of dangerous secrets as conversational memories
SENSITIVE_PATTERNS = [
    r"\bpass(?:word|phrase)\b",
    r"\bapi[_\-\s]*key\b",
    r"\bsecret[_\-\s]*token\b",
    r"\bauth[_\-\s]*token\b",
    r"\bcredit[_\-\s]*card\b",
    r"\bcvv\b",
    r"\bpin\s+number\b",
]


class MemoryManager:
    """Manages persistent personal memories, user preferences, and deterministic identity answers."""

    def __init__(self, store: Optional[LocalStore] = None):
        self.store = store or LocalStore()

    def get_preferred_name(self) -> Optional[str]:
        """Return the saved preferred user name or None."""
        mem = self.store.get_memory("user_name")
        return mem["value"] if mem else None

    def get_user_name(self) -> Optional[str]:
        """Alias for get_preferred_name."""
        return self.get_preferred_name()

    # =========================================================================
    # INTENT PARSER & EXECUTION
    # =========================================================================

    def try_handle_memory_command(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Check if query is an identity question, memory storage command, or memory retrieval command.
        Returns: (handled: bool, response: Optional[str])
        """
        q = query.strip()
        cleaned = re.sub(r"[.!?]+$", "", q).strip()
        q_lower = cleaned.lower()

        # 1. Direct Assistant Identity Queries
        if q_lower in [
            "what is your name",
            "what's your name",
            "who are you",
            "who are u",
            "tell me your name",
            "what are you called",
        ]:
            return True, get_assistant_identity_response()

        if q_lower in [
            "who created you",
            "who made you",
            "who is your creator",
            "who built you",
            "who developed you",
        ]:
            return True, get_creator_response()

        # 2. Direct User Name Queries
        if q_lower in [
            "what is my name",
            "what's my name",
            "who am i",
            "do you know my name",
            "tell me my name",
            "do you remember my name",
        ]:
            name_mem = self.store.get_memory("user_name")
            user_name = name_mem["value"] if name_mem else None
            return True, get_user_name_response(user_name)

        # 3. Update User Name
        update_name_match = re.search(
            r"^(?:update|change)\s+my\s+name\s+to\s+([A-Za-z\s'-]{2,40})$",
            cleaned,
            re.IGNORECASE,
        )
        if update_name_match:
            new_name = update_name_match.group(1).strip().title()
            self.store.set_memory(
                key="user_name",
                value=new_name,
                category="identity",
                source="user_explicit",
                confidence=1.0,
            )
            return True, f"I have updated your name to {new_name}."

        # 4. Explicit User Name Setting
        set_name_match = re.search(
            r"^(?:remember\s+that\s+)?(?:my\s+name\s+is|call\s+me)\s+([A-Za-z\s'-]{2,40})$",
            cleaned,
            re.IGNORECASE,
        )
        if not set_name_match:
            iam_match = re.search(r"^(?:remember\s+that\s+)?(?:i\s+am|i'm)\s+([A-Za-z'-]{2,20}(?:\s+[A-Za-z'-]{2,20})?)$", cleaned, re.IGNORECASE)
            if iam_match:
                cand = iam_match.group(1).strip().lower()
                non_names = {"a", "an", "the", "adam", "here", "ready", "fine", "good", "student", "developer", "engineer", "designer", "programmer"}
                if not any(w in cand.split() for w in non_names):
                    set_name_match = iam_match

        if set_name_match and not any(
            w in q_lower for w in ["adam", "assistant", "here", "ready", "going", "asking", "fine", "good"]
        ):
            name = set_name_match.group(1).strip().title()
            self.store.set_memory(
                key="user_name",
                value=name,
                category="identity",
                source="user_explicit",
                confidence=1.0,
            )
            return True, f"I will remember that your name is {name}."

        # 5. List / Inspect Memories
        if q_lower in [
            "what do you remember about me",
            "what do you remember",
            "what do you know about me",
            "show my memories",
            "list my memories",
            "what are my preferences",
            "show memory",
            "list memories",
        ]:
            return True, self._format_memories_list()

        # 6. Forget / Delete Memory Commands
        forget_match = re.search(
            r"^(?:forget|delete\s+memory|remove\s+memory)\s+(?:that\s+)?(.+)$",
            cleaned,
            re.IGNORECASE,
        )
        if forget_match:
            target = forget_match.group(1).strip().lower()
            return True, self._handle_forget(target)

        # 7. Explicit "Remember that ..." fact / preference storage
        remember_match = re.search(
            r"^(?:remember\s+that|remember)\s+(.+)$",
            cleaned,
            re.IGNORECASE,
        )
        if remember_match:
            fact_phrase = remember_match.group(1).strip()
            return True, self._handle_remember_statement(fact_phrase)

        return False, None

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _handle_remember_statement(self, phrase: str) -> str:
        """Parse and store an explicit user preference or fact with validation."""
        phrase_clean = phrase.strip()
        phrase_lower = phrase_clean.lower()

        # Ambiguous check: "remember this", "remember"
        if phrase_lower in ["this", "that", "it", "something", "everything"]:
            return "What specific fact or preference would you like me to remember?"

        # Security check: passwords / tokens
        for pat in SENSITIVE_PATTERNS:
            if re.search(pat, phrase_lower):
                return "⚠️ For your security, ADAM does not store passwords, secret tokens, or credentials in conversational memory."

        # Preferred language
        lang_match = re.search(r"(?:i\s+prefer|my\s+preferred\s+language\s+is)\s+(english|tamil|spanish|hindi|french|german|japanese|chinese|russian|italian|portuguese|korean|telugu|kannada|malayalam)(?:\s+responses|\s+language)?", phrase_lower)
        if lang_match:
            lang = lang_match.group(1).capitalize()
            self.store.set_memory(
                key="preferred_language",
                value=lang,
                category="preference",
                source="user_explicit",
            )
            return f"I will remember that you prefer {lang} responses."

        # Preferred name
        name_match = re.search(r"(?:my\s+name\s+is|call\s+me)\s+([A-Za-z\s'-]{2,40})", phrase_clean, re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip().title()
            self.store.set_memory(
                key="user_name",
                value=name,
                category="identity",
                source="user_explicit",
            )
            return f"I will remember that your name is {name}."

        # General fact
        words = re.sub(r"[^a-zA-Z0-9\s]", "", phrase_lower).split()
        key_slug = "_".join(words[:4]) if words else "custom_fact"
        self.store.set_memory(
            key=key_slug,
            value=phrase_clean,
            category="fact",
            source="user_explicit",
        )
        return f"Noted. I will remember that: \"{phrase_clean}\"."

    def _handle_forget(self, target: str) -> str:
        """Handle deletion of a specific memory key or concept."""
        clean_target = target.strip().lower()
        if "name" in clean_target:
            if self.store.delete_memory("user_name"):
                return "I have cleared your saved name from memory."
            return "I don't have a name saved for you."

        if "language" in clean_target:
            if self.store.delete_memory("preferred_language"):
                return "I have removed your language preference from memory."
            return "I don't have a preferred language saved for you."

        if clean_target in ["everything", "all", "all memories", "all facts"]:
            self.store.clear_memories()
            return "I have cleared all saved personal memories."

        # Try exact key or fuzzy match
        all_memories = self.store.list_memories()
        for m in all_memories:
            if clean_target in m["key"] or clean_target in m["value"].lower():
                self.store.delete_memory(m["key"])
                return f"I have forgotten: \"{m['value']}\"."

        return f"I couldn't find a memory matching '{target}' to remove."

    def _format_memories_list(self) -> str:
        """Format all stored memories for clear user inspection."""
        memories = self.store.list_memories()
        if not memories:
            return "I don't have any saved personal facts or preferences about you yet. Say e.g. \"My name is [name]\" or \"Remember that I prefer English\" to save facts."

        lines = ["🧠 **What I Remember About You:**"]
        categories = {}
        for m in memories:
            cat = m["category"].title()
            categories.setdefault(cat, []).append(m)

        for cat, items in sorted(categories.items()):
            lines.append(f"\n**{cat}:**")
            for item in items:
                key_display = item["key"].replace("_", " ").title()
                lines.append(f"• **{key_display}**: {item['value']}")

        return "\n".join(lines)

    def get_memory_context_prompt(self) -> str:
        """
        Generate a concise, structured facts block for injecting into the Ollama system prompt.
        Ensures the model stays grounded in confirmed facts without confusing identities.
        """
        memories = self.store.list_memories()
        if not memories:
            return ""

        lines = ["[Stored User Facts & Preferences - Strict Truth]"]
        for m in memories:
            k = m["key"].replace("_", " ").title()
            if k == "User Name":
                lines.append(f"- User's Preferred Name: {m['value']}")
            else:
                lines.append(f"- User {k}: {m['value']}")
        lines.append("Note: Assistant must never claim these user facts belong to ADAM.")
        return "\n".join(lines)
