import logging
import time
import re
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# CRITICAL SERVICES
from app.services.nlu_service import nlu_service, IntentType # <--- NEW
from app.services.pinecone_service import get_pinecone_service
from app.services.gemini_service import gemini_service
logger = logging.getLogger(__name__)

# ===================== DATA STRUCTURES =====================

class ConversationIntent(Enum):
    GREETING = "greeting"
    QUESTION = "question"
    FRUSTRATED = "frustrated"
    URGENT = "urgent"
    FAREWELL = "farewell"

@dataclass
class RAGResult:
    spoken_text: str
    fact_text: str
    intent: str
    entities: Dict[str, Any]
    sentiment: float
    confidence: float
    used_rag: bool

# ===================== ZERO-LATENCY "HyDE" LAYER =====================
# Competitors use this to "translate" bad voice transcripts into 
# formal database queries instantly (0ms latency).

_CONTRACTIONS = {
    r"\bwhat's\b": "what are",
    r"\bwhats\b": "what are",
    r"\bwhere's\b": "where is",
    r"\bwhen's\b": "when are",
    r"\bdon't\b": "do not",
    r"\bcan't\b": "cannot",
    r"\bi'm\b": "i am",
}

_QUERY_EXPANSIONS = {
    # Time/Schedule
    "timings": "business hours operating hours schedule opening time",
    "timing": "business hours operating hours schedule",
    "open": "business hours operating hours availability",
    "closed": "business hours operating hours closed",
    "schedule": "business hours appointment slots",
    
    # Cost/Money
    "price": "cost pricing rates fee charges",
    "cost": "price rates fee billing",
    "payments": "payment methods accepted cash upi credit card insurance",
    "pay": "payment methods options",
    
    # Action
    "appointment": "appointment booking consultation schedule visit",
    "book": "schedule appointment reservation",
    "contact": "phone number email address location",
    "phone": "contact number telephone mobile",
    "where": "location address directions",
}

def normalize_query(text: str) -> str:
    """
    Translates 'User English' to 'Database English'.
    Example: "whats ur timings" -> "what are business hours operating hours"
    """
    if not text:
        return ""
    
    s = text.lower().strip()
    
    # 1. Expand Contractions
    for patt, repl in _CONTRACTIONS.items():
        s = re.sub(patt, repl, s)
    
    # 2. Token Expansion (The "HyDE" Effect)
    tokens = s.split()
    out: List[str] = []
    
    i = 0
    while i < len(tokens):
        # Check bigrams (two words) first
        two_word = " ".join(tokens[i:i+2]) if i + 1 < len(tokens) else None
        
        if two_word and two_word in _QUERY_EXPANSIONS:
            out.extend(_QUERY_EXPANSIONS[two_word].split())
            i += 2
            continue
            
        # Check single words
        word = tokens[i]
        out.append(word)
        if word in _QUERY_EXPANSIONS:
            out.extend(_QUERY_EXPANSIONS[word].split())
        i += 1
        
    return " ".join(out)

# ===================== HELPERS =====================

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Robust JSON extractor for LLM responses"""
    if not text: return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start: return None
    blob = text[start:end+1]
    try:
        return json.loads(blob)
    except Exception:
        try:
            # Fix common LLM JSON errors (newlines, single quotes)
            safe = blob.replace("\n", " ").replace("'", '"')
            return json.loads(safe)
        except Exception:
            return None

def _rag_confidence(results: List[Dict[str, Any]]) -> float:
    """Calculates confidence based on Vector Similarity Scores"""
    if not results: return 0.0
    scores = [r.get("score", 0.0) for r in results if r.get("score") is not None]
    if not scores: return 0.0
    avg = sum(scores) / len(scores)
    # Normalize cosine similarity (-1 to 1) to confidence (0 to 1)
    return max(0.0, min(1.0, (avg + 1.0) / 2.0))

# ===================== THE ENGINE =====================

class RAGEngine:
    def __init__(self):
        """
        Competitor-Grade Configuration:
        - top_k=5: Fetch enough context to reduce hallucinations.
        - min_score=0.60: PERMISSIVE threshold. We let the LLM filter, not the DB.
        """
        self.top_k = 5
        self.min_score = 0.60 
        self.max_chunks = 3
        self.cache_ttl = 3600
        self.response_cache = {}
        logger.info(f"🚀 RAGEngine initialized (Threshold: {self.min_score})")

    def _get_cache_key(self, client_id: str, query: str) -> str:
        import hashlib
        normalized = normalize_query(query)
        key_str = f"{client_id}:{normalized}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[RAGResult]:
        if cache_key in self.response_cache:
            result, timestamp = self.response_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                logger.info(f"⚡ Cache hit")
                return result
            else:
                del self.response_cache[cache_key]
        return None

    def _set_cache(self, cache_key: str, result: RAGResult):
        self.response_cache[cache_key] = (result, time.time())
        # Simple LRU-like cleanup
        if len(self.response_cache) > 200:
            oldest = min(self.response_cache.keys(), key=lambda k: self.response_cache[k][1])
            del self.response_cache[oldest]
    #ef
    async def answer(self, client_id: str, query: str, session_context=None, language="en-IN") -> RAGResult:
        
        # 1. BRAIN: Analyze Intent
        nlu = await nlu_service.analyze(query, conversation_history=session_context)
        logger.info(f"🧠 Intent: {nlu.intent.value}")

        # 2. DIRECTOR: Choose the "Instruction" based on Intent
        # We do NOT return text here. We set the 'stage' for the LLM.
        
        system_instruction = ""
        context_data = ""
        used_rag = False

        if not nlu.topic_allowed:
            system_instruction = "The user asked about a forbidden topic. Politely decline to answer and steer back to business services."
            context_data = "Blocked Topic"

        elif nlu.intent == IntentType.GREETING:
            # NATURAL FIX: We tell LLM to *be* friendly, not what to say exactly.
            system_instruction = "You are a warm, professional assistant. The user just greeted you. Reply naturally and briefly. Vary your phrasing (e.g., 'Hi there', 'Good morning'). Ask how you can help."
            context_data = "User Greeting"

        elif nlu.intent == IntentType.CLOSING:
            system_instruction = "The user is saying goodbye. Reply politely and briefly. Wish them a good day."
            context_data = "User Closing"

        elif nlu.intent == IntentType.HUMAN_HANDOFF:
            system_instruction = "The user wants a real person. Politely confirm you are connecting them to a specialist immediately. Ask them to hold."
            context_data = "Handoff Request"

        else:
            # === RAG PATH (Questions, Billing, Booking) ===
            pinecone = get_pinecone_service()
            expanded = normalize_query(query)
            results = await pinecone.search_similar_chunks(client_id, expanded, self.top_k, self.min_score)
            
            if results:
                context_data = "\n".join([r['chunk_text'] for r in results])
                system_instruction = f"You are a helpful assistant. Answer the user's question strictly using this CONTEXT:\n{context_data}\nKeep it conversational but accurate."
                used_rag = True
            else:
                context_data = "No Info Found"
                system_instruction = "The user asked a question but we have no information in the database. Politely apologize and offer to connect them to a human."

        # 3. ACTOR: The LLM Generates the Speech
        # This single call handles EVERYTHING (Greetings, Goodbyes, and Data)
        try:
            raw_response = gemini_service.generate_response(
                prompt=f"USER SAYS: {query}", 
                system_message=system_instruction, 
                temperature=0.7 # Slight creativity for natural feel
            )
            
            # Clean up response (sometimes LLMs add "Assistant:" prefix)
            spoken_text = raw_response.replace("Assistant:", "").strip()
            
        except Exception:
            spoken_text = "I'm having a bit of trouble connecting. Could you say that again?"

        return RAGResult(
            spoken_text=spoken_text,
            fact_text=context_data[:50],
            intent=nlu.intent.value,
            entities=nlu.entities,
            sentiment=nlu.sentiment,
            confidence=nlu.confidence,
            used_rag=used_rag
        )


# SINGLETON EXPORT
rag_engine = RAGEngine()
