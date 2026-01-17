import logging
import json
import re
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import string # <--- Add this import at the top


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
    # CHANGED: Removed '^' and '$'. Added \b to match "Hi" inside "Hi there"
    IntentType.GREETING: [r"\b(hi|hello|hey|good morning|good evening|yo)\b"]
}

def _fast_reflex_check(text: str) -> Optional[NLUResult]:
    # SMART FIX: Remove punctuation! "Hi." becomes "hi"
    text_clean = text.lower().strip().translate(str.maketrans('', '', string.punctuation))
    
    for intent, patterns in _REFLEX_PATTERNS.items():
        for pattern in patterns:
            # Matches "hi there" because of \b check on cleaned text
            if re.search(pattern, text_clean):
                return NLUResult(intent, 1.0, 0.0, "low", {}, True, "Reflex Match")
    return None

# ===================== 3. THE ENTERPRISE BRAIN =====================

class NLUService:
    def __init__(self):
        self.CONFIDENCE_THRESHOLD = 0.60 
        
        # SMART FIX: A prompt that lists explicit examples to handle "Hii there"
        self.system_prompt = """
        You are the NLU Brain. CLASSIFY the user's intent.
        
        INTENT DEFINITIONS:
        - greeting: "hi", "hello", "good morning", "hii there", "hey"
        - booking: "schedule", "book appointment", "time for meeting"
        - cancellation: "cancel", "remove booking"
        - billing: "price", "cost", "how much"
        - human_handoff: "talk to agent", "real person"
        - question: General info queries ("what is X", "hours", "location")
        
        RULES:
        1. OUTPUT RAW JSON ONLY. Do NOT use Markdown (```).
        2. If the user says "Hii" or "Hii there", classify as "greeting".
        
        OUTPUT FORMAT:
        {"intent": "string", "confidence": 0.0-1.0, "urgency": "low|high", "topic_allowed": true}
        """
    #
    async def analyze(self, text: str, conversation_history: List[Dict[str, str]] = None) -> NLUResult:
        # 1. Reflex (Speed)
        reflex = _fast_reflex_check(text)
        if reflex:
            return reflex

        # 2. LLM Call
        try:
            full_prompt = f"{self.system_prompt}\nINPUT: \"{text}\""
            
            response_text = await gemini_service.generate_response_async(
                prompt=full_prompt, temperature=0.0, max_tokens=200
            )
            
            # SMART FIX: Log the RAW output so we can see why it failed before
            logger.info(f"🤖 RAW LLM OUTPUT: {response_text}")

            data = self._parse_json(response_text)
            
        except Exception as e:
            logger.error(f"❌ NLU Brain Failure: {e}")
            # If Brain dies, treat as a QUESTION so RAG can try to save it
            return NLUResult(IntentType.QUESTION, 0.5, 0.0, "low", {}, True, "Error Fallback")

        # 3. Governance (The Soft Fallback)
        confidence = float(data.get("confidence", 0.0))
        
        if confidence < self.CONFIDENCE_THRESHOLD:
            # SMART FIX: Don't block. Assume it's a question.
            logger.warning(f"⚠️ Low Confidence ({confidence}). Soft Fallback to QUESTION.")
            final_intent = IntentType.QUESTION
        else:
            intent_str = data.get("intent", "question").lower()
            try:
                final_intent = IntentType(intent_str)
            except ValueError:
                final_intent = IntentType.QUESTION

        return NLUResult(
            intent=final_intent,
            confidence=confidence,
            sentiment=0.0,
            urgency=data.get("urgency", "low"),
            entities=data.get("entities", {}),
            topic_allowed=data.get("topic_allowed", True),
            reasoning="Analyzed"
        )

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
