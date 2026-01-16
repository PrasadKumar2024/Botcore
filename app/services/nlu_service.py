import logging
import json
import re
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from app.services.gemini_service import gemini_service

logger = logging.getLogger(__name__)

# ===================== DATA STRUCTURES =====================

class IntentType(Enum):
    GREETING = "greeting"
    CLOSING = "closing"
    BOOKING = "booking"
    CANCELLATION = "cancellation"
    BILLING = "billing"
    SUPPORT = "support"
    HUMAN_HANDOFF = "human_handoff"
    QUESTION = "question" # General RAG fallback
    UNKNOWN = "unknown"

@dataclass
class NLUResult:
    intent: IntentType
    confidence: float
    sentiment: float # -1.0 to 1.0
    urgency: str # low, medium, high
    entities: Dict[str, Any]
    topic_allowed: bool

# ===================== REGEX REFLEX LAYER (0ms) =====================
# Handles high-frequency, low-complexity inputs instantly.

_REFLEX_PATTERNS = {
    IntentType.CLOSING: [
        r"\b(bye|goodbye|stop|end call|hang up)\b",
    ],
    IntentType.HUMAN_HANDOFF: [
        r"\b(human|agent|representative|operator|person)\b",
    ],
    IntentType.GREETING: [
        r"^(hi|hello|hey|good morning|good evening)$", # Strict start/end
    ]
}

def _fast_reflex_check(text: str) -> Optional[NLUResult]:
    text_lower = text.lower().strip()
    for intent, patterns in _REFLEX_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return NLUResult(
                    intent=intent,
                    confidence=1.0,
                    sentiment=0.0,
                    urgency="low",
                    entities={},
                    topic_allowed=True
                )
    return None

# ===================== LLM ROUTER LAYER (Smart) =====================

class NLUService:
    def __init__(self):
        # We use a dedicated System Prompt for the NLU task
        self.system_prompt = """
        You are the NLU (Natural Language Understanding) Brain for a Customer Service Voice Bot.
        
        YOUR JOB:
        Analyze the user's spoken text and output STRICT JSON.
        
        CLASSIFICATION RULES:
        1. intent: Choose one [greeting, closing, booking, cancellation, billing, support, human_handoff, question, unknown]
        2. confidence: 0.0 to 1.0 (How sure are you?)
        3. sentiment: -1.0 (Angry) to 1.0 (Happy)
        4. urgency: [low, medium, high]
        5. entities: Extract named entities (dates, names, account_ids)
        6. topic_allowed: boolean. FALSE if user asks about politics, religion, or competitors. TRUE otherwise.
        
        INPUT: "{text}"
        OUTPUT JSON:
        """

    async def analyze(self, text: str, business_context: str = "") -> NLUResult:
        """
        Main Entry Point.
        1. Checks Reflex Layer.
        2. If no reflex, calls LLM for deep analysis.
        """
        # 1. Reflex Check (0ms)
        reflex = _fast_reflex_check(text)
        if reflex:
            logger.info(f"⚡ NLU Reflex Hit: {reflex.intent}")
            return reflex

        # 2. LLM Analysis (300ms)
        try:
            # We assume gemini_service is initialized and working
            prompt = self.system_prompt.replace("{text}", text)
            if business_context:
                prompt += f"\nBusiness Context: {business_context}"

            # Use low temp for deterministic JSON
            response_text = await gemini_service.generate_response_async(
                prompt=prompt,
                temperature=0.0, 
                max_tokens=200
            )
            
            # Parse JSON safely
            data = self._parse_json(response_text)
            
            return NLUResult(
                intent=IntentType(data.get("intent", "question")),
                confidence=float(data.get("confidence", 0.0)),
                sentiment=float(data.get("sentiment", 0.0)),
                urgency=data.get("urgency", "low"),
                entities=data.get("entities", {}),
                topic_allowed=data.get("topic_allowed", True)
            )

        except Exception as e:
            logger.error(f"❌ NLU LLM Failed: {e}")
            # Fallback to generic question if brain fails
            return NLUResult(
                intent=IntentType.QUESTION,
                confidence=0.0,
                sentiment=0.0,
                urgency="low",
                entities={},
                topic_allowed=True
            )

    def _parse_json(self, text: str) -> Dict[str, Any]:
        try:
            # Strip markdown code blocks if Gemini adds them
            clean_text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except json.JSONDecodeError:
            logger.warning(f"⚠️ NLU Malformed JSON: {text}")
            return {}

# Singleton
nlu_service = NLUService()
