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
        
        
    async def answer(self, client_id: str, query: str, session_context=None, language="en-IN") -> RAGResult:
        
        # 1. CALL THE BRAIN (nlu_service, NOT gemini_service)
        nlu = await nlu_service.analyze(query)
        logger.info(f"🧠 Intent: {nlu.intent.value}")

        # 2. GUARDRAILS
        if not nlu.topic_allowed:
            return RAGResult("I can only discuss our services.", "", "off_topic", {}, 0.0, 1.0, False)

        # 3. ROUTING
        if nlu.intent == IntentType.GREETING:
            return RAGResult("Hello! How can I help you?", "Greeting", "greeting", {}, 0.5, 1.0, False)
            
        if nlu.intent == IntentType.HUMAN_HANDOFF:
             return RAGResult("Please hold for an agent.", "Handoff", "handoff", {}, 0.0, 1.0, False)

        # 4. RAG EXECUTION
        pinecone = get_pinecone_service()
        expanded_query = normalize_query(query)
        
        # Search (Try Expanded -> Then Raw)
        results = await pinecone.search_similar_chunks(client_id, expanded_query, self.top_k, self.min_score)
        if not results:
             results = await pinecone.search_similar_chunks(client_id, query, self.top_k, self.min_score)

        if results:
            context = "\n".join([r['chunk_text'] for r in results])
        else:
            context = "NO INFO FOUND."

        # Generation (Using Gemini Service directly)
        sys_prompt = f"You are a helpful assistant. Answer using this CONTEXT:\n{context}"
        raw = gemini_service.generate_response(f"Query: {query}", system_message=sys_prompt, temperature=0.1)
        
        # Simple extraction (assuming gemini returns clean text if JSON fails)
        spoken = raw if "{" not in raw else _extract_json(raw).get("spoken", "I don't have that info.")

        return RAGResult(spoken, context[:50], nlu.intent.value, nlu.entities, nlu.sentiment, 0.9, True)


# SINGLETON EXPORT
rag_engine = RAGEngine()
