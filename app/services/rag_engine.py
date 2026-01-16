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

    async def answer(
        self,
        *,
        client_id: str,
        query: str,
        session_context: Optional[List[Dict[str, Any]]] = None,
        language: str = "en-IN",
    ) -> RAGResult:
        start_ts = time.time()
        
        # 1. ROUTER: Detect Intent
        intent, sentiment = self._detect_intent_and_emotion(query)
        
        # --- FAST PATH: GREETINGS ---
        if intent == ConversationIntent.GREETING:
            import random
            greetings = [
                "Hello! How can I assist you today?",
                "Hi there! What can I do for you?",
                "Good day! I'm here to help."
            ]
            return RAGResult(
                spoken_text=random.choice(greetings),
                fact_text="Greeting", intent="greeting", entities={}, 
                sentiment=0.5, confidence=1.0, used_rag=False
            )

        # 2. CHECK CACHE
        cache_key = self._get_cache_key(client_id, query)
        cached = self._check_cache(cache_key)
        if cached: return cached

        # 3. EXPANSION: The "HyDE" Effect
        # Translate "timings" -> "business hours" to match database vectors
        expanded_query = normalize_query(query)
        logger.info(f"🔍 RAG Search | Raw: '{query}' | Expanded: '{expanded_query}'")

        # 4. SEARCH: Strict Client Isolation
        pinecone = get_pinecone_service()
        results = await pinecone.search_similar_chunks(
            client_id=client_id,
            query=expanded_query,   # Use the optimized query
            top_k=self.top_k,
            min_score=self.min_score # Permissive threshold (0.60)
        )

        # 5. RETRIEVAL LOGIC
        if results:
            # We found relevant chunks!
            chunk_texts = [f"INFO: {r.get('chunk_text','')}" for r in results[:self.max_chunks]]
            context_block = "\n\n".join(chunk_texts)
        else:
            # Fallback: Try searching the RAW query if expanded failed
            logger.info("⚠️ Expanded query yielded 0 results. Trying raw query...")
            results_raw = await pinecone.search_similar_chunks(
                client_id=client_id,
                query=query,
                top_k=self.top_k,
                min_score=self.min_score
            )
            if results_raw:
                chunk_texts = [f"INFO: {r.get('chunk_text','')}" for r in results_raw[:self.max_chunks]]
                context_block = "\n\n".join(chunk_texts)
                results = results_raw
            else:
                context_block = "NO INFORMATION FOUND."

        # 6. GENERATION: Strict Business Guardrails
        system_prompt = (
            "You are a helpful Voice Assistant for a business.\n"
            "INSTRUCTIONS:\n"
            "1. Answer the User Question based ONLY on the provided INFO.\n"
            "2. If the INFO is 'NO INFORMATION FOUND' or irrelevant, say: 'I don't have that information. I can only answer questions about our services.'\n"
            "3. Keep answers SHORT (1-2 sentences) and conversational for voice.\n"
            "4. Never invent facts, prices, or hours.\n\n"
            "OUTPUT JSON: "
            '{"spoken": "The natural spoken answer", "confidence": 0.0 to 1.0, "sentiment": -1.0 to 1.0}'
        )
        
        user_prompt = f"INFO:\n{context_block}\n\nUSER QUESTION: {query}\n\nReturn JSON."

        try:
            raw_response = gemini_service.generate_response(
                prompt=user_prompt,
                system_message=system_prompt,
                temperature=0.1, # Low temp for factual accuracy
                max_tokens=200
            )
            parsed = _extract_json(raw_response)
        except Exception as e:
            logger.error(f"❌ LLM Generation Failed: {e}")
            parsed = None

        # 7. RESPONSE FORMATTING
        if not parsed:
            # Fallback if LLM crashes
            final_spoken = "I'm having trouble retrieving that information. Could you ask again?"
            confidence = 0.0
        else:
            final_spoken = parsed.get("spoken", "I don't have that information.")
            confidence = float(parsed.get("confidence", 0.8))
            
            # Sanity Check: If Context was empty, force confidence down
            if context_block == "NO INFORMATION FOUND.":
                confidence = 0.0
                # Ensure the LLM didn't hallucinate an answer despite no info
                if len(final_spoken) > 20 and "don't have" not in final_spoken:
                    final_spoken = "I'm sorry, I don't have any information about that in my records."

        # 8. CONSTRUCT RESULT
        rag_result = RAGResult(
            spoken_text=final_spoken,
            fact_text=context_block[:100], # Store brief context for debugging
            intent="question",
            entities={},
            sentiment=0.0,
            confidence=confidence,
            used_rag=True
        )

        self._set_cache(cache_key, rag_result)
        logger.info(f"✅ RAG Answer ({confidence*100:.0f}% conf) in {(time.time()-start_ts)*1000:.0f}ms")
        return rag_result

# SINGLETON EXPORT
rag_engine = RAGEngine()
