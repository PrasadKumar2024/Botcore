import logging
import json
import re
from typing import Dict, Any, Optional, List
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
    QUESTION = "question"         
    OFF_TOPIC = "off_topic"
    UNKNOWN = "unknown"

@dataclass
class NLUResult:
    intent: IntentType
    confidence: float
    entities: Dict[str, Any]
    topic_allowed: bool
    data_source: str    # <--- NEW: 'live', 'static', or 'none'
    reasoning: str 

class NLUService:
    def __init__(self):
        self.CONFIDENCE_THRESHOLD = 0.60
        
        self.system_prompt = """
        You are the Routing Intelligence.
        
        TASK: Analyze the user's input and output a JSON object.
        
        1. "intent": 
           - 'greeting' (Hello, Hi)
           - 'booking' (I want to book)
           - 'question' (Information requests)
           - 'off_topic' (Jokes, Politics, Math)
           
        2. "data_source" (CRITICAL DECISION):
           - 'live': Queries about STAFF, AVAILABILITY, ROSTER, STATUS. 
             (Examples: "Who is in?", "Is Suresh free?", "Can I see anyone?")
           - 'static': Queries about POLICIES, LOCATION, REFUNDS.
             (Examples: "Where are you?", "What is the fee?")
           - 'none': Greetings, closings, or off-topic.
           
        3. "entities":
           - Extract "name" if a specific person is mentioned.
           - Extract "date" or "time" for bookings.
        
        OUTPUT FORMAT: {"intent": "...", "data_source": "...", "entities": {}, "confidence": 1.0}
        """

    def _safe_json_parse(self, text: str) -> Dict[str, Any]:
        """Robust parser that handles markdown ticks and messy JSON."""
        try:
            # 1. Strip Markdown
            clean_text = text.replace("```json", "").replace("```", "").strip()
            
            # 2. Try Standard Load
            return json.loads(clean_text)
        except json.JSONDecodeError:
            logger.warning(f"⚠️ JSON Parse Failed. Raw Text: {text}")
            # 3. Fallback: Try to find the first '{' and last '}'
            try:
                start = clean_text.find("{")
                end = clean_text.rfind("}")
                if start != -1 and end != -1:
                    return json.loads(clean_text[start:end+1])
            except:
                pass
            return {} # Final fallback

    async def analyze(self, text: str, conversation_history: List = None) -> NLUResult:
        # [Insert Tier 1 Regex Logic Here if you use it]
        
        try:
            # CALL GEMINI
            response_text = await gemini_service.generate_response_async(
                prompt=f"{self.system_prompt}\nINPUT: \"{text}\"", 
                temperature=0.0, 
                max_tokens=300
            )
            
            data = self._safe_json_parse(response_text)
            
            # DEFAULT VALUES (If JSON is empty)
            intent_str = data.get("intent", "question")
            data_source = data.get("data_source", "static")
            
            # Validate Intent Enum
            try:
                final_intent = IntentType(intent_str)
            except ValueError:
                final_intent = IntentType.QUESTION

            result = NLUResult(
                intent=final_intent,
                confidence=data.get("confidence", 0.0),
                entities=data.get("entities", {}),
                topic_allowed=True, # You can add logic to set this to False if intent is off_topic
                data_source=data_source,
                reasoning="Tier 3: Smart Analysis"
            )
            
            logger.info(f"🧠 NLU Analysis: Intent={result.intent.value} | Source={result.data_source} | Entities={result.entities}")
            return result

        except Exception as e:
            logger.error(f"❌ NLU Critical Crash: {e}")
            # SAFE FALLBACK: Treat as a general question so the bot doesn't die
            return NLUResult(IntentType.QUESTION, 0.5, {}, True, "static", "Error Recovery")
    

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
