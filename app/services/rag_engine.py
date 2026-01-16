import logging
import time
import re
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# CRITICAL SERVICES
from app.services.pinecone_service import get_pinecone_service
from app.services.gemini_service import gemini_service, IntentType
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

    def _detect_intent_and_emotion(self, query: str) -> Tuple[ConversationIntent, float]:
        """
        Fast Heuristic Router (0ms Latency).
        Decides if we need RAG or just a quick reply.
        """
        q = query.lower()
        
        # 1. Frustration Check
        bad_words = ['angry', 'upset', 'hate', 'stupid', 'broken', 'fail', 'terrible']
        if any(w in q for w in bad_words):
            return ConversationIntent.FRUSTRATED, -0.8

        # 2. Greeting Check
        # Strict check: "Hello" is a greeting. "Hello can I..." is a Question.
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good evening', 'hi there']
        words = q.split()
        if len(words) <= 3 and any(w in q for w in greetings):
            return ConversationIntent.GREETING, 0.5
            
        return ConversationIntent.QUESTION, 0.0
        
        
    async def answer(
        self,
        *,
        client_id: str,
        query: str,
        session_context: Optional[List[Dict[str, Any]]] = None,
        language: str = "en-IN",
    ) -> RAGResult:
        
        # A. CALL THE BRAIN
        # -------------------------------------------
        nlu = await gemini_service.analyze_user_input(query, "the company")

        # B. HANDLE GUARDRAILS
        # -------------------------------------------
        if not nlu.topic_allowed:
            return RAGResult(
                spoken_text="I can only discuss topics related to our services. How can I help with that?",
                fact_text="Off-topic blocked", intent="off_topic", entities={}, 
                sentiment=0.0, confidence=1.0, used_rag=False
            )

        # C. HANDLE SPECIFIC ROUTING
        # -------------------------------------------
        if nlu.intent == IntentType.GREETING:
            import random
            greetings = ["Hello! How can I assist you?", "Hi there! I'm ready to help."]
            return RAGResult(
                spoken_text=random.choice(greetings),
                fact_text="Greeting", intent="greeting", entities={}, 
                sentiment=0.5, confidence=1.0, used_rag=False
            )

        if nlu.intent == IntentType.HUMAN_HANDOFF:
             return RAGResult(
                spoken_text="I'll connect you with a specialist immediately. Please hold.",
                fact_text="Handoff", intent="handoff", entities={}, 
                sentiment=0.0, confidence=1.0, used_rag=False
            )

        # D. HANDLE RAG (Questions/Billing/Support)
        # -------------------------------------------
        
        # Use NLU metadata to adjust the RAG persona
        is_urgent = nlu.urgency == "high" or nlu.intent == IntentType.CANCELLATION
        
        # 1. Search Logic (Same as before)
        pinecone = get_pinecone_service()
        expanded_query = normalize_query(query)
        results = await pinecone.search_similar_chunks(client_id, expanded_query, self.top_k, self.min_score)
        
        if not results:
             results = await pinecone.search_similar_chunks(client_id, query, self.top_k, self.min_score)

        if results:
            context_block = "\n".join([r['chunk_text'] for r in results])
        else:
            context_block = "NO INFORMATION FOUND."

        # 2. Generation Logic (Persona Injection)
        if is_urgent:
            persona = "You are an Urgent Support Agent. Answer immediately and concisely. Apologize if needed."
        else:
            persona = "You are a helpful Assistant. Answer conversationally."

        system_prompt = f"{persona}\n\nCONTEXT:\n{context_block}\n\nStrictly answer based on CONTEXT."
        
        # ... (Call Gemini Generate) ...
        # (This part of your code remains standard)
        
        # Return result using the INTENT found by the brain
        return RAGResult(
            spoken_text="...result from gemini...", # placeholder for your generation code
            fact_text=context_block[:50],
            intent=nlu.intent.value, # Use the string value of the Enum
            entities=nlu.entities,
            sentiment=nlu.sentiment,
            confidence=0.9,
            used_rag=True
        )


# SINGLETON EXPORT
rag_engine = RAGEngine()
