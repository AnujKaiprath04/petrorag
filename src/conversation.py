"""
PetroRAG Conversational RAG & Context Carryover Engine (Module 2.29)
Maintains multi-turn dialogue state, accumulates technical entities across turns,
and resolves elliptical pronouns (e.g. 'its', 'the unit') to preserve domain context.
"""

from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.core.logging import logger
from src.query.entities import RuleBasedEntityExtractor


class DialogueTurn(BaseModel):
    """An individual query-response exchange."""
    turn_id: int
    user_query: str
    assistant_answer: str
    resolved_query: str
    entities: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DialogueSession(BaseModel):
    """Stateful technical conversation thread."""
    session_id: str
    history: List[DialogueTurn] = Field(default_factory=list)
    active_equipment_ids: List[str] = Field(default_factory=list)
    active_fields: List[str] = Field(default_factory=list)
    active_wells: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConversationManager:
    """
    Manages active dialogue sessions and resolves context carryover for multi-turn QA.
    """

    PRONOUN_PATTERNS = [
        r"\b(?:it|its|this|that|the unit|the equipment|the machine|the pump|the compressor|the separator|the valve)\b"
    ]

    def __init__(self, max_turns_retained: int = 10):
        self.max_turns_retained = max_turns_retained
        self.sessions: Dict[str, DialogueSession] = {}
        self.entity_extractor = RuleBasedEntityExtractor()

    def get_or_create_session(self, session_id: str) -> DialogueSession:
        """Retrieves active session or initializes a new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = DialogueSession(session_id=session_id)
        return self.sessions[session_id]

    def resolve_followup_query(self, session_id: str, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Resolves elliptical pronouns in follow-up queries by binding active entities
        from previous dialogue turns.
        Example: 'What is its trip limit?' + active 'C-101' -> 'What is C-101 trip limit?'
        """
        session = self.get_or_create_session(session_id)
        current_entities = self.entity_extractor.extract(query)
        resolved = query.strip()

        # If current query lacks equipment IDs, check if previous turns have active equipment
        if not current_entities.equipment_ids and session.active_equipment_ids:
            latest_equipment = session.active_equipment_ids[-1]

            # Check if query contains pronouns like "its", "the unit", etc.
            has_pronoun = any(re.search(pat, query, re.IGNORECASE) for pat in self.PRONOUN_PATTERNS)

            if has_pronoun:
                # Replace pronouns or append equipment tag for precise retrieval
                resolved = re.sub(r"\b(?:its|the unit's|the equipment's)\b", f"{latest_equipment}'s", resolved, flags=re.IGNORECASE)
                resolved = re.sub(r"\b(?:it|the unit|the equipment|the machine)\b", latest_equipment, resolved, flags=re.IGNORECASE)
            else:
                # Query has no equipment and no pronoun, but is in session context: append reference
                resolved = f"{resolved} for {latest_equipment}"

            logger.info(f"Resolved contextual query for session {session_id}: '{query}' -> '{resolved}'")

        return resolved, current_entities.model_dump()

    def record_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        resolved_query: str,
        entities: Optional[Dict[str, Any]] = None
    ) -> DialogueTurn:
        """
        Appends completed exchange to session memory and updates active entity state.
        """
        session = self.get_or_create_session(session_id)
        turn_num = len(session.history) + 1

        ent_dict = entities or {}
        # Accumulate newly discovered equipment tags
        new_equip = ent_dict.get("equipment_ids", [])
        for eq in new_equip:
            if eq not in session.active_equipment_ids:
                session.active_equipment_ids.append(eq)

        turn = DialogueTurn(
            turn_id=turn_num,
            user_query=user_query,
            assistant_answer=assistant_answer,
            resolved_query=resolved_query,
            entities=ent_dict
        )
        session.history.append(turn)

        # Enforce turn window limit
        if len(session.history) > self.max_turns_retained:
            session.history = session.history[-self.max_turns_retained:]

        return turn

    def get_chat_history_for_prompt(
        self,
        session_id: str,
        max_turns: int = 5
    ) -> List[Dict[str, str]]:
        """
        Formats recent session turns into role-based messages for LLM chat prompt.
        """
        session = self.get_or_create_session(session_id)
        recent_turns = session.history[-max_turns:] if session.history else []

        messages: List[Dict[str, str]] = []
        for turn in recent_turns:
            messages.append({"role": "user", "content": turn.user_query})
            messages.append({"role": "assistant", "content": turn.assistant_answer})

        return messages
