import logging
import time
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from app.services.pinecone_service import pinecone_service
from app.services.gemini_service import gemini_service

logger = logging.getLogger(__name__)

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

    return " ".join(out)


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


def _rag_confidence(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    scores = [r.get("score", 0.0) for r in results if r.get("score") is not None]
    if not scores:
        return 0.0
    avg = sum(scores) / len(scores)
    # heuristic normalization (matches old behavior)
    return max(0.0, min(1.0, avg + 0.5))


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

    def __init__(self):
        self.max_chunks = 4
        self.top_k = 6
        self.min_score = -1.0

    async def answer(
        self,
        *,
        client_id: str,
        query: str,
        session_context: Optional[List[Dict[str, Any]]] = None,
        language: str = "en-IN",
    ) -> RAGResult:

        start_ts = time.time()

        # 1. Normalize query
        normalized = normalize_query(query)

        # 2. Retrieve KB context (CLIENT-SCOPED)
        try:
            results = await pinecone_service.search_similar_chunks(
                client_id=client_id,
                query=normalized or query,
                top_k=self.top_k,
                min_score=self.min_score,
            )
        except Exception as e:
            logger.exception("Pinecone search failed: %s", e)
            results = []

        if not results:
            # No RAG hit → explicit signal to orchestration layer
            return RAGResult(
                spoken_text="",
                fact_text="",
                intent="",
                entities={},
                sentiment=0.0,
                confidence=0.0,
                used_rag=False,
            )

        # 3. Build context block
        context_text = "\n\n".join(
            f"{r.get('source','')}: {r.get('chunk_text','')}"
            for r in results[:self.max_chunks]
        )

        # 4. Conversation memory (short-term only)
        convo_block = ""
        if session_context:
            lines = []
            for m in session_context[-6:]:
                role = "User" if m.get("role") == "user" else "Assistant"
                lines.append(f"{role}: {m.get('text','')}")
            convo_block = "\n".join(lines) + "\n\n"

        # 5. Structured Gemini prompt
        system_prompt = (
            "You are a professional, warm voice assistant. "
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
            "- If context is insufficient, say you don’t have that info"
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

        return RAGResult(
            spoken_text=spoken or fact,
            fact_text=fact,
            intent=intent,
            entities=entities,
            sentiment=sentiment,
            confidence=confidence,
            used_rag=True,
        )


# -----------------------------
# Singleton
# -----------------------------

rag_engine = RAGEngine()
