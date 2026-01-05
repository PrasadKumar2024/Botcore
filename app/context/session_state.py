
import time
import uuid
import asyncio
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Deque, Optional, List


# ==============================
# Dialogue State Machines
# ==============================

class DialogueFlow(str, Enum):
    NONE = "none"
    BOOKING = "booking"
    BILLING = "billing"
    CANCELLATION = "cancellation"
    ESCALATION = "escalation"


class BookingState(str, Enum):
    START = "start"
    ASK_DATE = "ask_date"
    ASK_TIME = "ask_time"
    CONFIRM = "confirm"
    COMPLETED = "completed"
    FAILED = "failed"


class BillingState(str, Enum):
    START = "start"
    IDENTIFY_INVOICE = "identify_invoice"
    EXPLAIN_CHARGES = "explain_charges"
    RESOLVE = "resolve"
    ESCALATE = "escalate"
    COMPLETED = "completed"


# ==============================
# Cancellation & Concurrency
# ==============================

class CancellationToken:
    """
    Thread + asyncio safe cancellation primitive.
    Used for:
      - TTS cancellation
      - LLM streaming cancellation
      - STT restart
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def reset(self) -> None:
        self._event.clear()


# ==============================
# Conversation Memory Entry
# ==============================

@dataclass(slots=True)
class ConversationTurn:
    role: str                    # "user" | "assistant"
    text: str
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    sentiment: float = 0.0
    confidence: float = 0.0
    ts: float = field(default_factory=lambda: time.time())


# ==============================
# Dialogue Context (FSM container)
# ==============================

@dataclass
class DialogueContext:
    flow: DialogueFlow = DialogueFlow.NONE
    state: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.flow = DialogueFlow.NONE
        self.state = None
        self.data.clear()


# ==============================
# Session State (CORE OBJECT)
# ==============================

class SessionState:
    """
    Single-session authoritative state container.

    Owned by:
      - voice.py (orchestration)
      - gemini_service.py (NLU decisions)

    Never shared across sessions.
    """

    __slots__ = (
        "session_id",
        "client_id",
        "created_at",
        "language",
        "is_bot_speaking",
        "memory",
        "dialogue",
        "tts_cancel_token",
        "llm_cancel_token",
        "stt_restart_token",
        "speaking_rate",
        "lock",
    )

    def __init__(
        self,
        client_id: str,
        language: str = "en-US",
        memory_limit: int = 20,
    ) -> None:
        self.session_id: str = f"s_{uuid.uuid4().hex}"
        self.client_id: str = client_id
        self.created_at: float = time.time()

        # Runtime flags
        self.language: str = language
        self.is_bot_speaking: bool = False

        # Short-term memory (hot context)
        self.memory: Deque[ConversationTurn] = deque(maxlen=memory_limit)

        # Dialogue FSM
        self.dialogue: DialogueContext = DialogueContext()

        # Cancellation tokens
        self.tts_cancel_token = CancellationToken()
        self.llm_cancel_token = CancellationToken()
        self.stt_restart_token = CancellationToken()

        # Concurrency protection
        self.lock = threading.RLock()
        self.speaking_rate = 1.0

    # ==========================
    # Memory Management
    # ==========================

    def add_turn(
        self,
        role: str,
        text: str,
        *,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
        sentiment: float = 0.0,
        confidence: float = 0.0,
    ) -> None:
        with self.lock:
            self.memory.append(
                ConversationTurn(
                    role=role,
                    text=text,
                    intent=intent,
                    entities=entities or {},
                    sentiment=sentiment,
                    confidence=confidence,
                )
            )

    def recent_turns(self, n: int = 6) -> List[ConversationTurn]:
        with self.lock:
            return list(self.memory)[-n:]

    # ==========================
    # Dialogue Flow Control
    # ==========================

    def start_flow(self, flow: DialogueFlow) -> None:
        with self.lock:
            self.dialogue.flow = flow
            if flow == DialogueFlow.BOOKING:
                self.dialogue.state = BookingState.START
            elif flow == DialogueFlow.BILLING:
                self.dialogue.state = BillingState.START
            else:
                self.dialogue.state = None
            self.dialogue.data.clear()

    def update_state(self, new_state: str) -> None:
        with self.lock:
            self.dialogue.state = new_state

    def end_flow(self) -> None:
        with self.lock:
            self.dialogue.reset()

    # ==========================
    # Barge-in & Cancellation
    # ==========================

    def on_barge_in(self) -> None:
        """
        Called when user speaks while bot is speaking.
        Must be fast, safe, idempotent.
        """
        with self.lock:
            self.is_bot_speaking = False
            self.tts_cancel_token.cancel()
            self.llm_cancel_token.cancel()

    def reset_cancellations(self) -> None:
        with self.lock:
            self.tts_cancel_token.reset()
            self.llm_cancel_token.reset()

    # ==========================
    # Language / STT control
    # ==========================

    def set_language(self, language: str) -> None:
        with self.lock:
            if language != self.language:
                self.language = language
                self.stt_restart_token.cancel()

    # ==========================
    # Redis Sync Hooks (future)
    # ==========================

    def to_redis_payload(self) -> Dict[str, Any]:
        """
        Safe serialization for Redis.
        """
        with self.lock:
            return {
                "session_id": self.session_id,
                "client_id": self.client_id,
                "language": self.language,
                "dialogue": {
                    "flow": self.dialogue.flow.value,
                    "state": self.dialogue.state,
                    "data": self.dialogue.data,
                },
                "memory": [
                    {
                        "role": t.role,
                        "text": t.text,
                        "intent": t.intent,
                        "entities": t.entities,
                        "sentiment": t.sentiment,
                        "confidence": t.confidence,
                        "ts": t.ts,
                    }
                    for t in self.memory
                ],
                "created_at": self.created_at,
            }

    @classmethod
    def from_redis_payload(cls, payload: Dict[str, Any]) -> "SessionState":
        """
        Restore session from Redis snapshot.
        """
        state = cls(
            client_id=payload["client_id"],
            language=payload.get("language", "en-US"),
        )
        state.session_id = payload["session_id"]
        state.created_at = payload.get("created_at", time.time())

        dlg = payload.get("dialogue", {})
        state.dialogue.flow = DialogueFlow(dlg.get("flow", "none"))
        state.dialogue.state = dlg.get("state")
        state.dialogue.data = dlg.get("data", {})

        for t in payload.get("memory", []):
            state.memory.append(
                ConversationTurn(
                    role=t["role"],
                    text=t["text"],
                    intent=t.get("intent"),
                    entities=t.get("entities", {}),
                    sentiment=t.get("sentiment", 0.0),
                    confidence=t.get("confidence", 0.0),
                    ts=t.get("ts", time.time()),
                )
            )

        return state
