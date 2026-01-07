import logging
import time
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from app.services.pinecone_service import pinecone_service
from app.services.pinecone_service import get_pinecone_service
from app.services.gemini_service import gemini_service
from sentence_transformers import CrossEncoder
from enum import Enum

logger = logging.getLogger(__name__)

# Intent classification for dialogue management
class ConversationIntent(Enum):
    GREETING = "greeting"
    QUESTION = "question"
    FRUSTRATED = "frustrated"
    URGENT = "urgent"
    SATISFIED = "satisfied"
    CONFUSED = "confused"
    FAREWELL = "farewell"
# -----------------------------
# Public result contract
# -----------------------------

@dataclass
class RAGResult:
    spoken_text: str
    fact_text: str
    intent: str
    entities: Dict[str, Any]
    sentiment: float
    confidence: float
    used_rag: bool


# -----------------------------
# Query normalization (EXTRACTED from old system)
# -----------------------------

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

# -----------------------------
# Helpers
# -----------------------------

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

def _get_cache_key(self, client_id: str, query: str) -> str:
    """Generate cache key from client_id and normalized query"""
    import hashlib
    normalized = normalize_query(query)
    key_str = f"{client_id}:{normalized}"
    return hashlib.md5(key_str.encode()).hexdigest()

def _check_cache(self, cache_key: str) -> Optional[RAGResult]:
    """Check if cached response is still valid"""
    if cache_key in self.response_cache:
        result, timestamp = self.response_cache[cache_key]
        if time.time() - timestamp < self.cache_ttl:
            logger.info(f"Cache hit for key: {cache_key[:8]}...")
            return result
        else:
            del self.response_cache[cache_key]
    return None

def _set_cache(self, cache_key: str, result: RAGResult):
    """Store result in cache with size limit"""
    self.response_cache[cache_key] = (result, time.time())
    
    if len(self.response_cache) > 100:
        oldest_key = min(
            self.response_cache.keys(),
            key=lambda k: self.response_cache[k][1]
        )
        del self.response_cache[oldest_key]

async def _rerank_results(
    self,
    query: str,
    results: List[Dict[str, Any]],
    top_n: int = 4
) -> List[Dict[str, Any]]:
    """
    Re-rank search results using cross-encoder for semantic relevance.
    Returns top_n most relevant chunks.
    """
    if not self.use_reranker or not results:
        return results[:top_n]
    
    try:
        pairs = [
            (query, r.get('chunk_text', ''))
            for r in results
        ]
        
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            self.reranker.predict,
            pairs
        )
        
        for i, result in enumerate(results):
            result['rerank_score'] = float(scores[i])
        
        reranked = sorted(
            results,
            key=lambda x: x.get('rerank_score', 0.0),
            reverse=True
        )
        
        logger.info(
            f"Re-ranked {len(results)} results, "
            f"top score: {reranked[0].get('rerank_score', 0):.3f}"
        )
        
        return reranked[:top_n]
        
    except Exception as e:
        logger.exception(f"Re-ranking failed: {e}")
        return results[:top_n]

def _detect_intent_and_emotion(
    self,
    query: str,
    session_context: Optional[List[Dict[str, Any]]] = None
) -> Tuple[ConversationIntent, float, int]:
    """
    Detect user intent, emotion score, and urgency level.
    Returns tuple of intent, emotion score, and urgency level.
    """
    query_lower = query.lower()
    
    urgency_words = ['urgent', 'hurry', 'quick', 'quickly', 'immediately', 'now', 'asap', 'fast']
    urgency_level = sum(1 for word in urgency_words if word in query_lower)
    urgency_level = min(urgency_level, 3)
    
    frustration_words = ['frustrated', 'angry', 'annoyed', 'upset', 'disappointed', 
                         'waste', 'useless', 'terrible', 'awful', 'horrible']
    is_frustrated = any(word in query_lower for word in frustration_words)
    
    confusion_words = ['confused', "don't understand", "dont understand", 'what do you mean', 
                       'unclear', 'explain again', 'not clear']
    is_confused = any(phrase in query_lower for phrase in confusion_words)
    
    question_words = ['what', 'when', 'where', 'who', 'why', 'how', '?']
    is_question = any(word in query_lower for word in question_words)
    
    greeting_words = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
    is_greeting = any(word in query_lower for word in greeting_words)
    
    farewell_words = ['bye', 'goodbye', 'thank you', 'thanks', "that's all", 'thats all']
    is_farewell = any(word in query_lower for word in farewell_words)
    
    if is_greeting:
        intent = ConversationIntent.GREETING
        emotion = 0.5
    elif is_frustrated:
        intent = ConversationIntent.FRUSTRATED
        emotion = -0.7
    elif is_confused:
        intent = ConversationIntent.CONFUSED
        emotion = -0.3
    elif urgency_level >= 2:
        intent = ConversationIntent.URGENT
        emotion = 0.0
    elif is_farewell:
        intent = ConversationIntent.FAREWELL
        emotion = 0.6
    elif is_question:
        intent = ConversationIntent.QUESTION
        emotion = 0.0
    else:
        intent = ConversationIntent.QUESTION
        emotion = 0.0
    
    return intent, emotion, urgency_level
    
def _rag_confidence(results: List[Dict[str, Any]]) -> float:
    """
    Calculate confidence from both embedding score and rerank score.
    Prioritizes rerank scores when available.
    """
    if not results:
        return 0.0
    
    rerank_scores = [
        r.get("rerank_score", 0.0) 
        for r in results 
        if r.get("rerank_score") is not None
    ]
    
    if rerank_scores:
        avg_rerank = sum(rerank_scores) / len(rerank_scores)
        normalized = avg_rerank / 10.0
        return max(0.0, min(1.0, normalized))
    
    embed_scores = [
        r.get("score", 0.0) 
        for r in results 
        if r.get("score") is not None
    ]
    
    if not embed_scores:
        return 0.0
    
    avg = sum(embed_scores) / len(embed_scores)
    return max(0.0, min(1.0, (avg + 1.0) / 2.0))

# -----------------------------
# Main RAG Engine
# -----------------------------

class RAGEngine:
    """
    Client-aware Retrieval-Augmented Generation engine.

    Responsibilities:
    - Normalize query
    - Retrieve client-scoped context from Pinecone
    - Build grounded Gemini prompt
    - Parse structured JSON response
    - Fuse confidence (LLM + RAG)
    - Provide safe fallback
    """
    def _get_cache_key(self, client_id: str, query: str) -> str:
        """Generates a simple unique key for caching responses."""
        import hashlib
        # Create a unique hash based on user ID and their question
        return hashlib.md5(f"{client_id}:{query}".encode()).hexdigest()

    async def answer(
        self,
        *,
        client_id: str,
        query: str,
        session_context: Optional[List[Dict[str, Any]]] = None,
        language: str = "en-IN",
    ) -> RAGResult:

        start_ts = time.time()
    
        cache_key = self._get_cache_key(client_id, query)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result

    # 1. Normalize query
        normalized = normalize_query(query)
    
    # 1.5. Run intent detection and Pinecone search in parallel
        intent_task = asyncio.create_task(
            asyncio.to_thread(
                self._detect_intent_and_emotion,
                query=query,
                session_context=session_context
            )
        )
    
        pinecone_task = asyncio.create_task(
            get_pinecone_service.search_similar_chunks(
                client_id=client_id,
                query=normalized or query,
                top_k=self.top_k,
                min_score=self.min_score,
            )
        )
    
        try:
            intent_result, results = await asyncio.gather(
                intent_task,
                pinecone_task,
                return_exceptions=True
            )
        
            if isinstance(intent_result, Exception):
                logger.exception("Intent detection failed")
                intent_detected, emotion_score, urgency_level = (
                    ConversationIntent.QUESTION, 0.0, 0
                )
            else:
                intent_detected, emotion_score, urgency_level = intent_result
        
            if isinstance(results, Exception):
                logger.exception("Pinecone search failed")
                results = []
            
        except Exception as e:
            logger.exception("Parallel operations failed: %s", e)
            results = []
            intent_detected, emotion_score, urgency_level = (
                ConversationIntent.QUESTION, 0.0, 0
            )
    
        logger.info(
            f"Intent detected: {intent_detected.value}, "
            f"emotion: {emotion_score:.2f}, urgency: {urgency_level}"
        )

        if not results:
            return RAGResult(
                spoken_text="",
                fact_text="",
                intent="",
                entities={},
                sentiment=0.0,
                confidence=0.0,
                used_rag=False,
            )

    # 2.5. Re-rank results with cross-encoder
        results = await self._rerank_results(
            query=query,
            results=results,
            top_n=self.max_chunks
        )

    # 3. Build context block
        context_text = "\n\n".join(
            f"{r.get('source','')}: {r.get('chunk_text','')}"
            for r in results
        )

    # 4. Conversation memory
        convo_block = ""
        if session_context:
            lines = []
            for m in session_context[-6:]:
                role = "User" if m.get("role") == "user" else "Assistant"
                lines.append(f"{role}: {m.get('text','')}")
            convo_block = "\n".join(lines) + "\n\n"

    # 5. Intent-aware system prompt
        base_instructions = "You are a professional, warm voice assistant."
    
        if intent_detected == ConversationIntent.FRUSTRATED:
            base_instructions += (
                "\n**CRITICAL: The user is frustrated. "
                "Start with a genuine apology, acknowledge their concern, "
                "then provide a clear, direct answer.**"
            )
        elif intent_detected == ConversationIntent.URGENT:
            base_instructions += (
                "\n**The user needs urgent information. "
                "Be concise and direct. Skip pleasantries.**"
            )
        elif intent_detected == ConversationIntent.CONFUSED:
            base_instructions += (
                "\n**The user is confused. "
                "Explain step-by-step in simpler terms. "
                "Ask if they need clarification.**"
            )
        elif intent_detected == ConversationIntent.GREETING:
            base_instructions += (
                "\n**This is a greeting. Respond warmly and briefly. "
                "Ask how you can help.**"
            )
    
        system_prompt = (
            f"{base_instructions}\n\n"
            "Use ONLY the CONTEXT to answer factual questions.\n\n"
            "Return EXACT JSON with keys:\n"
            "{"
            "\"intent\":\"\","
            "\"entities\":{},"
            "\"confidence\":0.0,"
            "\"sentiment\":0.0,"
            "\"fact\":\"\","
            "\"spoken\":\"\""
            "}\n\n"
            "Rules:\n"
            "- spoken: 2–3 short, phone-friendly sentences\n"
            "- Use 'we' when speaking for the business\n"
            "- Offer a next step when appropriate\n"
            "- If context is insufficient, say you don't have that info"
        )

        user_prompt = (
            f"{convo_block}"
            f"CONTEXT:\n{context_text}\n\n"
            f"USER QUESTION:\n{query}\n\n"
            f"Return JSON only."
        )

        # 6. Gemini call (non-streaming, grounded)
        try:
            raw = gemini_service.generate_response(
                prompt=user_prompt,
                system_message=system_prompt,
                temperature=0.15,
                max_tokens=350,
            )
        except Exception as e:
            logger.exception("Gemini RAG call failed: %s", e)
            raw = ""

        parsed = _extract_json(raw)
        rag_conf = _rag_confidence(results)

        if not parsed:
            # Hard fallback (still grounded)
            spoken = (
                "I found some related information, but I want to be sure. "
                "Could you please clarify your question?"
            )
            return RAGResult(
                spoken_text=spoken,
                fact_text="",
                intent="rag_fallback",
                entities={},
                sentiment=0.0,
                confidence=rag_conf,
                used_rag=True,
            )

        # 7. Final fusion
        intent = parsed.get("intent", "")
        entities = parsed.get("entities", {}) or {}
        sentiment = float(parsed.get("sentiment", 0.0) or 0.0)
        fact = (parsed.get("fact") or "").strip()
        spoken = (parsed.get("spoken") or "").strip()
        llm_conf = float(parsed.get("confidence", rag_conf) or rag_conf)

        confidence = max(rag_conf, llm_conf)

        logger.info(
            "RAG answered | client=%s intent=%s conf=%.2f latency=%.0fms",
            client_id,
            intent,
            confidence,
            (time.time() - start_ts) * 1000,
        )

        final_result = RAGResult(
            spoken_text=spoken or fact,
            fact_text=fact,
            intent=intent,
            entities=entities,
            sentiment=sentiment,
            confidence=confidence,
            used_rag=True,
        )
    
        self._set_cache(cache_key, final_result)
    
        return final_result

# -----------------------------
# Singleton
# -----------------------------

rag_engine = RAGEngine()
