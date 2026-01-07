import logging
import time
import re
import json
import asyncio  # ← CRITICAL MISSING IMPORT
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from app.services.pinecone_service import get_pinecone_service
from app.services.gemini_service import gemini_service
from enum import Enum

logger = logging.getLogger(__name__)

class ConversationIntent(Enum):
    GREETING = "greeting"
    QUESTION = "question"
    FRUSTRATED = "frustrated"
    URGENT = "urgent"
    SATISFIED = "satisfied"
    CONFUSED = "confused"
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

# Keep these as GLOBAL helpers (they don't need self)
_CONTRACTIONS = {
    r"\bwhat's\b": "what are",
    r"\bwhats\b": "what are",
    r"\bwhere's\b": "where is",
    r"\bwhen's\b": "when are",
    r"\bdon't\b": "do not",
    r"\bcan't\b": "cannot",
}

_QUERY_EXPANSIONS = {
    "timings": "business hours operating hours schedule",
    "timing": "business hours operating hours schedule",
    "open": "business hours operating hours",
    "closed": "business hours operating hours",
    "appointment": "appointment booking consultation",
    "doctor": "doctor physician consultant",
    "payments": "payment methods accepted cash upi card",
    "phone number": "contact number telephone",
}

def normalize_query(text: str) -> str:
    if not text:
        return ""
    s = text.lower().strip()
    urgency_markers = []
    for word in ['urgent', 'quickly', 'immediately', 'asap', 'hurry', 'fast']:
        if word in s:
            urgency_markers.append(word)
    for patt, repl in _CONTRACTIONS.items():
        s = re.sub(patt, repl, s)
    tokens = s.split()
    out: List[str] = []
    i = 0
    while i < len(tokens):
        two = " ".join(tokens[i:i+2]) if i + 1 < len(tokens) else None
        if two and two in _QUERY_EXPANSIONS:
            out.extend(_QUERY_EXPANSIONS[two].split())
            i += 2
            continue
        t = tokens[i]
        out.append(t)
        if t in _QUERY_EXPANSIONS:
            out.extend(_QUERY_EXPANSIONS[t].split())
        i += 1
    result = " ".join(out)
    if urgency_markers:
        result = " ".join(urgency_markers) + " " + result
    return result

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    blob = text[start:end+1]
    try:
        return json.loads(blob)
    except Exception:
        try:
            safe = blob.replace("\n", " ").replace("'", '"')
            return json.loads(safe)
        except Exception:
            return None

def _rag_confidence(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    embed_scores = [r.get("score", 0.0) for r in results if r.get("score") is not None]
    if not embed_scores:
        return 0.0
    avg = sum(embed_scores) / len(embed_scores)
    return max(0.0, min(1.0, (avg + 1.0) / 2.0))

# ===================== RAGEngine CLASS =====================
class RAGEngine:
    def __init__(self):
        """Initialize with safe defaults"""
        self.top_k = 5
        self.min_score = 0.7
        self.max_chunks = 3
        self.use_reranker = False  # Disable for stability
        self.cache_ttl = 3600
        self.response_cache = {}
        logger.info("RAGEngine initialized")

    def _get_cache_key(self, client_id: str, query: str) -> str:
        import hashlib
        normalized = normalize_query(query)
        key_str = f"{client_id}:{normalized}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[RAGResult]:
        if cache_key in self.response_cache:
            result, timestamp = self.response_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                logger.info(f"Cache hit: {cache_key[:8]}")
                return result
            else:
                del self.response_cache[cache_key]
        return None

    def _set_cache(self, cache_key: str, result: RAGResult):
        self.response_cache[cache_key] = (result, time.time())
        if len(self.response_cache) > 100:
            oldest = min(self.response_cache.keys(), 
                        key=lambda k: self.response_cache[k][1])
            del self.response_cache[oldest]

    def _detect_intent_and_emotion(
        self, 
        query: str, 
        session_context: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[ConversationIntent, float, int]:
        query_lower = query.lower()
        urgency_words = ['urgent', 'hurry', 'quick', 'quickly', 'immediately', 'now', 'asap']
        urgency_level = sum(1 for word in urgency_words if word in query_lower)
        urgency_level = min(urgency_level, 3)
        
        frustration_words = ['frustrated', 'angry', 'annoyed', 'upset', 'terrible']
        is_frustrated = any(word in query_lower for word in frustration_words)
        
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', '?']
        is_question = any(word in query_lower for word in question_words)
        
        greeting_words = ['hello', 'hi', 'hey', 'good morning']
        is_greeting = any(word in query_lower for word in greeting_words)
        
        if is_greeting:
            return ConversationIntent.GREETING, 0.5, 0
        elif is_frustrated:
            return ConversationIntent.FRUSTRATED, -0.7, urgency_level
        elif urgency_level >= 2:
            return ConversationIntent.URGENT, 0.0, urgency_level
        elif is_question:
            return ConversationIntent.QUESTION, 0.0, urgency_level
        else:
            return ConversationIntent.QUESTION, 0.0, 0

    async def answer(
        self,
        *,
        client_id: str,
        query: str,
        session_context: Optional[List[Dict[str, Any]]] = None,
        language: str = "en-IN",
    ) -> RAGResult:
        
        start_ts = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(client_id, query)
        cached = self._check_cache(cache_key)
        if cached:
            return cached
        
        # Normalize query
        normalized = normalize_query(query)
        
        # Parallel operations
        intent_task = asyncio.create_task(
            asyncio.to_thread(
                self._detect_intent_and_emotion,
                query, session_context
            )
        )
        
        pinecone_task = asyncio.create_task(
            get_pinecone_service().search_similar_chunks(
                client_id=client_id,
                query=normalized or query,
                top_k=self.top_k,
                min_score=self.min_score,
            )
        )
        
        try:
            intent_result, results = await asyncio.gather(
                intent_task, pinecone_task, return_exceptions=True
            )
            
            if isinstance(intent_result, Exception):
                logger.error(f"Intent failed: {intent_result}")
                intent_detected, emotion_score, urgency_level = (
                    ConversationIntent.QUESTION, 0.0, 0
                )
            else:
                intent_detected, emotion_score, urgency_level = intent_result
            
            if isinstance(results, Exception):
                logger.error(f"Pinecone failed: {results}")
                results = []
                
        except Exception as e:
            logger.exception(f"Parallel ops failed: {e}")
            results = []
            intent_detected, emotion_score, urgency_level = (
                ConversationIntent.QUESTION, 0.0, 0
            )
        
        logger.info(f"Intent: {intent_detected.value}, emotion: {emotion_score:.2f}")
        
        # Handle no results
        if not results:
            fallback = RAGResult(
                spoken_text="I'm sorry, I don't have enough information to answer that. Could you rephrase your question?",
                fact_text="",
                intent="no_results",
                entities={},
                sentiment=0.0,
                confidence=0.0,
                used_rag=False,
            )
            self._set_cache(cache_key, fallback)
            return fallback
        
        # Build context
        context_text = "\n\n".join(
            f"{r.get('source','')}: {r.get('chunk_text','')}"
            for r in results[:self.max_chunks]
        )
        
        # Build prompt
        system_prompt = (
            "You are a professional voice assistant. "
            "Answer in 2-3 short sentences suitable for speaking. "
            "Return JSON only: "
            '{\"intent\":\"\",\"entities\":{},\"confidence\":0.0,'
            '\"sentiment\":0.0,\"fact\":\"\",\"spoken\":\"\"}'
        )
        
        user_prompt = f"CONTEXT:\n{context_text}\n\nQUESTION:\n{query}\n\nReturn JSON only."
        
        # Call Gemini
        try:
            raw = gemini_service.generate_response(
                prompt=user_prompt,
                system_message=system_prompt,
                temperature=0.15,
                max_tokens=300,
            )
        except Exception as e:
            logger.exception(f"Gemini failed: {e}")
            raw = ""
        
        parsed = _extract_json(raw)
        rag_conf = _rag_confidence(results)
        
        if not parsed:
            fallback = RAGResult(
                spoken_text="I found some information but need clarification. Could you rephrase?",
                fact_text="",
                intent="parse_failed",
                entities={},
                sentiment=0.0,
                confidence=rag_conf,
                used_rag=True,
            )
            self._set_cache(cache_key, fallback)
            return fallback
        
        # Extract result
        result = RAGResult(
            spoken_text=parsed.get("spoken", "") or parsed.get("fact", ""),
            fact_text=parsed.get("fact", ""),
            intent=parsed.get("intent", ""),
            entities=parsed.get("entities", {}),
            sentiment=float(parsed.get("sentiment", 0.0)),
            confidence=max(rag_conf, float(parsed.get("confidence", rag_conf))),
            used_rag=True,
        )
        
        self._set_cache(cache_key, result)
        
        logger.info(f"RAG answered in {(time.time()-start_ts)*1000:.0f}ms")
        return result

# Singleton
rag_engine = RAGEngine()
