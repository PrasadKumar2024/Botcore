import logging
import json
import re
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Use your existing service
from app.services.gemini_service import gemini_service

logger = logging.getLogger(__name__)

# ===================== 1. DEFINITIONS =====================

class IntentType(str, Enum):
    GREETING = "greeting"
    CLOSING = "closing"
    BOOKING = "booking"
    CANCELLATION = "cancellation"
    BILLING = "billing"
    SUPPORT = "support"
    HUMAN_HANDOFF = "human_handoff"
    QUESTION = "question"         # Valid query for RAG
    CLARIFICATION = "clarification" # Low confidence fallback
    UNKNOWN = "unknown"

@dataclass
class NLUResult:
    intent: IntentType
    confidence: float
    sentiment: float      # -1.0 (Negative) to 1.0 (Positive)
    urgency: str          # low, medium, high
    entities: Dict[str, Any]
    topic_allowed: bool   # False if political/toxic

# ===================== 2. REFLEX LAYER (0ms) =====================
# Hard-coded rules that override AI for safety/speed.

_REFLEX_PATTERNS = {
    IntentType.CLOSING: [r"\b(bye|goodbye|stop|end call|hang up|quit)\b"],
    IntentType.HUMAN_HANDOFF: [r"\b(human|agent|representative|operator|person)\b"],
    IntentType.GREETING: [r"^(hi|hello|hey|good morning|good evening)$"]
}

def _fast_reflex_check(text: str) -> Optional[NLUResult]:
    text_lower = text.lower().strip()
    for intent, patterns in _REFLEX_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return NLUResult(intent, 1.0, 0.0, "low", {}, True)
    return None

# ===================== 3. THE ENTERPRISE BRAIN =====================

class NLUService:
    def __init__(self):
        # CONFIDENCE GATE: Below this, we force clarification.
        self.CONFIDENCE_THRESHOLD = 0.65
        
        self.system_prompt = """
        You are an Enterprise NLU Router. Analyze user input for a Business Voice Bot.
        
        RULES:
        1. CLASSIFY INTENT: [booking, cancellation, billing, support, question, greeting, closing]
        2. DETECT ENTITIES: Extract dates, times, money, names.
        3. CHECK SAFETY: set "topic_allowed": false if input is sexual, political, or hate speech.
        
        INPUT: "{text}"
        
        OUTPUT STRICT JSON:
        {"intent": "string", "confidence": 0.0-1.0, "sentiment": -1.0-1.0, "urgency": "low|medium|high", "entities": {}, "topic_allowed": true}
        """

    async def analyze(self, text: str) -> NLUResult:
        """
        Multi-Stage Pipeline: Reflex -> LLM -> Governance
        """
        # --- STAGE 1: REFLEX (Speed) ---
        reflex = _fast_reflex_check(text)
        if reflex:
            logger.info(f"⚡ Reflex: {reflex.intent.value}")
            return reflex

        # --- STAGE 2: COGNITION (LLM) ---
        try:
            # We use the async wrapper you added to gemini_service
            prompt = self.system_prompt.replace("{text}", text)
            response_text = await gemini_service.generate_response_async(
                prompt=prompt, temperature=0.0, max_tokens=200
            )
            raw_data = self._parse_json(response_text)
        except Exception as e:
            logger.error(f"❌ NLU Brain Dead: {e}")
            return self._fallback_result()

        # --- STAGE 3: GOVERNANCE (Logic Gates) ---
        return self._apply_governance_logic(raw_data)

    def _apply_governance_logic(self, data: Dict[str, Any]) -> NLUResult:
        """Applies Confidence Gating and Fallback Logic."""
        
        # 1. Safety Gate
        if not data.get("topic_allowed", True):
            logger.warning("🛡️ NLU Safety Block Triggered")
            return NLUResult(IntentType.UNKNOWN, 1.0, 0.0, "low", {}, False)

        # 2. Extract & Validate Confidence
        try:
            confidence = float(data.get("confidence", 0.0))
        except:
            confidence = 0.0

        # 3. Confidence Gate
        if confidence < self.CONFIDENCE_THRESHOLD:
            logger.info(f"📉 Low Confidence ({confidence:.2f}) -> Clarification")
            return NLUResult(
                intent=IntentType.CLARIFICATION, 
                confidence=confidence,
                sentiment=0.0, urgency="low", entities={}, topic_allowed=True
            )

        # 4. Intent Mapping
        intent_str = data.get("intent", "question").lower()
        try:
            final_intent = IntentType(intent_str)
        except ValueError:
            final_intent = IntentType.QUESTION # Default fallback

        return NLUResult(
            intent=final_intent,
            confidence=confidence,
            sentiment=float(data.get("sentiment", 0.0)),
            urgency=data.get("urgency", "low"),
            entities=data.get("entities", {}),
            topic_allowed=True
        )

    def _parse_json(self, text: str) -> Dict[str, Any]:
        try:
            clean = text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean)
        except:
            return {}

    def _fallback_result(self) -> NLUResult:
        return NLUResult(IntentType.CLARIFICATION, 0.0, 0.0, "low", {}, True)

# Singleton
nlu_service = NLUService()
