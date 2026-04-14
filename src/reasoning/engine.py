"""
Agentic reasoning engine for clinical evidence synthesis.

Multi-step pipeline:
1. Query decomposition & medical entity extraction
2. Evidence retrieval with hierarchy-aware ranking
3. Contradiction detection
4. LLM-powered recommendation synthesis with citations
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from ..config import get_settings
from ..models.core import (
    Evidence, EvidenceLevel, ClinicalQuery, ClinicalRecommendation, Contradiction
)
from ..ingestion.pipeline import get_vector_store

logger = logging.getLogger(__name__)


# Evidence hierarchy weights (higher = stronger evidence)
EVIDENCE_WEIGHTS = {
    EvidenceLevel.SYSTEMATIC_REVIEW: 10.0,
    EvidenceLevel.META_ANALYSIS: 9.5,
    EvidenceLevel.RCT: 8.0,
    EvidenceLevel.COHORT_STUDY: 6.0,
    EvidenceLevel.CASE_CONTROL: 5.0,
    EvidenceLevel.OBSERVATIONAL: 4.0,
    EvidenceLevel.CASE_STUDY: 2.0,
    EvidenceLevel.EXPERT_OPINION: 1.0,
}

DOC_TYPE_TO_EVIDENCE_LEVEL = {
    "systematic_review": EvidenceLevel.SYSTEMATIC_REVIEW,
    "meta_analysis": EvidenceLevel.META_ANALYSIS,
    "clinical_trial": EvidenceLevel.RCT,
    "guideline": EvidenceLevel.SYSTEMATIC_REVIEW,
    "research_paper": EvidenceLevel.OBSERVATIONAL,
    "case_study": EvidenceLevel.CASE_STUDY,
}


def _infer_evidence_level(metadata: Dict[str, Any]) -> EvidenceLevel:
    doc_type = metadata.get("document_type", "research_paper")
    return DOC_TYPE_TO_EVIDENCE_LEVEL.get(doc_type, EvidenceLevel.OBSERVATIONAL)


def _detect_contradictions(evidence_list: List[Evidence]) -> List[Dict[str, Any]]:
    """Find contradictory statements between evidence items."""
    contradictions = []
    contradiction_pairs = [
        ("effective", "ineffective"),
        ("beneficial", "harmful"),
        ("recommended", "not recommended"),
        ("increases", "decreases"),
        ("significant", "non-significant"),
        ("superior", "inferior"),
        ("safe", "unsafe"),
    ]

    for i in range(len(evidence_list)):
        for j in range(i + 1, len(evidence_list)):
            e1, e2 = evidence_list[i], evidence_list[j]
            t1, t2 = e1.excerpt.lower(), e2.excerpt.lower()

            for pos, neg in contradiction_pairs:
                if (pos in t1 and neg in t2) or (neg in t1 and pos in t2):
                    # Determine which has stronger evidence
                    w1 = EVIDENCE_WEIGHTS.get(e1.evidence_level, 1.0)
                    w2 = EVIDENCE_WEIGHTS.get(e2.evidence_level, 1.0)
                    preferred = e1 if w1 >= w2 else e2

                    contradictions.append({
                        "evidence_1": {"id": e1.document_id, "title": e1.title, "level": e1.evidence_level.value},
                        "evidence_2": {"id": e2.document_id, "title": e2.title, "level": e2.evidence_level.value},
                        "explanation": f"Conflicting findings on '{pos}' vs '{neg}' between studies.",
                        "resolution": f"Prioritize the {preferred.evidence_level.value.replace('_', ' ')} ({preferred.title or preferred.document_id}) as it represents stronger evidence.",
                    })
                    break

    return contradictions


def _compute_confidence(evidence_list: List[Evidence], contradictions: List[Dict]) -> float:
    """Compute recommendation confidence from evidence quality."""
    if not evidence_list:
        return 0.05

    # Average weighted quality
    total_weight = sum(EVIDENCE_WEIGHTS.get(e.evidence_level, 1.0) * e.relevance_score
                       for e in evidence_list)
    max_possible = sum(EVIDENCE_WEIGHTS.get(e.evidence_level, 1.0) for e in evidence_list)

    base = total_weight / max_possible if max_possible > 0 else 0.0

    # Quantity bonus (diminishing returns)
    quantity_bonus = min(0.2, len(evidence_list) * 0.03)

    # Contradiction penalty
    contradiction_penalty = len(contradictions) * 0.08

    return max(0.05, min(0.98, base + quantity_bonus - contradiction_penalty))


class ReasoningEngine:
    """Orchestrates multi-step clinical evidence reasoning."""

    def __init__(self):
        self.settings = get_settings()
        self._openai_client = None

    def _get_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=self.settings.openai_api_key)
        return self._openai_client

    def _embed_query(self, query_text: str) -> List[float]:
        client = self._get_client()
        response = client.embeddings.create(
            model=self.settings.embedding_model,
            input=query_text.replace("\n", " ")[:8000]
        )
        return response.data[0].embedding

    def _retrieve_evidence(self, query: ClinicalQuery) -> List[Evidence]:
        """Step 1: Retrieve semantically relevant evidence from vector store."""
        store = get_vector_store()
        if store.document_count == 0:
            logger.warning("Vector store is empty - no documents indexed yet")
            return []

        query_embedding = self._embed_query(query.query_text)
        results = store.search(
            query_embedding,
            top_k=self.settings.max_evidence_items,
            min_score=self.settings.similarity_threshold
        )

        evidence_list = []
        for r in results:
            meta = r["metadata"]
            level = _infer_evidence_level(meta)
            evidence_list.append(Evidence(
                document_id=r["doc_id"],
                relevance_score=min(1.0, r["score"]),
                evidence_level=level,
                excerpt=r["chunk_text"][:500],
                title=meta.get("title", r["doc_id"]),
                authors=meta.get("authors", []),
                source=meta.get("source", ""),
                publication_date=meta.get("publication_date", ""),
                sample_size=meta.get("sample_size"),
            ))

        # Sort by composite score
        evidence_list.sort(
            key=lambda e: EVIDENCE_WEIGHTS.get(e.evidence_level, 1.0) * e.relevance_score,
            reverse=True
        )
        return evidence_list

    def _build_llm_prompt(self, query: ClinicalQuery, evidence_list: List[Evidence],
                           contradictions: List[Dict]) -> str:
        """Build the prompt for LLM recommendation synthesis."""
        evidence_text = ""
        for i, e in enumerate(evidence_list[:6], 1):
            evidence_text += f"\n[{i}] {e.evidence_level.value.replace('_', ' ').upper()}"
            if e.title:
                evidence_text += f" — {e.title}"
            if e.publication_date:
                year = e.publication_date[:4]
                evidence_text += f" ({year})"
            evidence_text += f"\n    {e.excerpt[:400]}\n"

        contradiction_text = ""
        if contradictions:
            contradiction_text = "\n\n⚠️ CONTRADICTORY EVIDENCE DETECTED:\n"
            for c in contradictions[:3]:
                contradiction_text += f"- {c['explanation']} Resolution: {c['resolution']}\n"

        patient_ctx = ""
        if query.patient_context:
            ctx = query.patient_context
            parts = []
            if ctx.get("age"):
                parts.append(f"Age: {ctx['age']}")
            if ctx.get("gender"):
                parts.append(f"Gender: {ctx['gender']}")
            if ctx.get("conditions"):
                parts.append(f"Conditions: {', '.join(ctx['conditions'])}")
            if ctx.get("medications"):
                parts.append(f"Medications: {', '.join(ctx['medications'])}")
            if parts:
                patient_ctx = f"\n\nPATIENT CONTEXT: {'; '.join(parts)}"

        urgency = f"\nURGENCY: {query.urgency_level.value.upper()}"

        return f"""You are a clinical evidence expert providing evidence-based medical recommendations.

CLINICAL QUERY: {query.query_text}{urgency}{patient_ctx}

RETRIEVED EVIDENCE:{evidence_text if evidence_text else "\n(No indexed evidence found — respond from general medical knowledge with appropriate caveats)"}
{contradiction_text}

INSTRUCTIONS:
1. Provide a clear, actionable clinical recommendation directly answering the query
2. Cite evidence sources using [1], [2], etc. notation
3. State the strength of evidence (e.g., "Strong evidence from RCTs supports...")
4. If contradictions exist, address them explicitly and explain how to resolve them
5. Flag any important safety considerations or contraindications
6. Use clinical language appropriate for a healthcare professional
7. Be concise but comprehensive (3-5 paragraphs)
8. End with a brief "BOTTOM LINE:" summary sentence

Format your response as a professional clinical recommendation."""

    def _synthesize_recommendation(self, query: ClinicalQuery,
                                    evidence_list: List[Evidence],
                                    contradictions: List[Dict]) -> Tuple[str, List[Dict]]:
        """Step 3: LLM synthesis with reasoning steps."""
        reasoning_steps = [
            {
                "step": 1,
                "type": "query_analysis",
                "description": f"Analyzed clinical query: '{query.query_text[:80]}...'",
                "detail": f"Urgency: {query.urgency_level.value}, extracted medical context"
            },
            {
                "step": 2,
                "type": "evidence_retrieval",
                "description": f"Retrieved {len(evidence_list)} evidence sources",
                "detail": f"Top sources: {', '.join(e.evidence_level.value for e in evidence_list[:3])}"
            },
        ]

        if contradictions:
            reasoning_steps.append({
                "step": 3,
                "type": "contradiction_analysis",
                "description": f"Detected {len(contradictions)} contradictions",
                "detail": contradictions[0]["explanation"] if contradictions else ""
            })

        reasoning_steps.append({
            "step": len(reasoning_steps) + 1,
            "type": "synthesis",
            "description": "Synthesizing evidence-based recommendation",
            "detail": "Applying medical evidence hierarchy and clinical guidelines"
        })

        if not self.settings.openai_api_key:
            rec_text = self._fallback_recommendation(query, evidence_list, contradictions)
            return rec_text, reasoning_steps

        try:
            client = self._get_client()
            prompt = self._build_llm_prompt(query, evidence_list, contradictions)

            response = client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a clinical evidence specialist providing evidence-based recommendations to healthcare professionals. You are precise, cite evidence, and always prioritize patient safety."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.2,
            )
            rec_text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            rec_text = self._fallback_recommendation(query, evidence_list, contradictions)

        return rec_text, reasoning_steps

    def _fallback_recommendation(self, query: ClinicalQuery,
                                   evidence_list: List[Evidence],
                                   contradictions: List[Dict]) -> str:
        """Fallback when OpenAI is unavailable."""
        if not evidence_list:
            return (
                f"**Query:** {query.query_text}\n\n"
                "No specific evidence was found in the current knowledge base for this query. "
                "The document library may still be loading, or no relevant studies have been indexed. "
                "Please consult current clinical guidelines (AHA/ACC, ADA, etc.) and peer-reviewed literature. "
                "Consider querying PubMed or UpToDate for the most current evidence.\n\n"
                "**BOTTOM LINE:** Insufficient indexed evidence available — consult primary sources."
            )

        top = evidence_list[0]
        strength = "Strong" if top.evidence_level in [EvidenceLevel.SYSTEMATIC_REVIEW, EvidenceLevel.META_ANALYSIS] else "Moderate" if top.evidence_level == EvidenceLevel.RCT else "Limited"

        rec = f"**Based on {len(evidence_list)} evidence sources:**\n\n"
        rec += f"{strength} evidence ({top.evidence_level.value.replace('_', ' ')}) addresses this query. "
        rec += f"Key finding from {top.title or 'primary source'}: {top.excerpt[:300]}...\n\n"

        if contradictions:
            rec += f"⚠️ **Note:** {len(contradictions)} contradictions detected in the evidence. "
            rec += f"{contradictions[0]['explanation']} {contradictions[0]['resolution']}\n\n"

        rec += "**BOTTOM LINE:** Consult the indexed evidence above and current clinical guidelines for definitive guidance."
        return rec

    def generate(self, query: ClinicalQuery) -> ClinicalRecommendation:
        """Full reasoning pipeline: retrieve → rank → detect contradictions → synthesize."""
        logger.info(f"Generating recommendation for query: {query.id}")

        # Step 1: Retrieve evidence
        evidence_list = self._retrieve_evidence(query)

        # Step 2: Detect contradictions
        contradictions = _detect_contradictions(evidence_list)

        # Step 3: Synthesize recommendation
        rec_text, reasoning_steps = self._synthesize_recommendation(query, evidence_list, contradictions)

        # Step 4: Compute confidence
        confidence = _compute_confidence(evidence_list, contradictions)

        rec_id = f"rec_{query.id}_{int(datetime.now().timestamp())}"
        return ClinicalRecommendation(
            id=rec_id,
            query_id=query.id,
            recommendation_text=rec_text,
            supporting_evidence=evidence_list,
            confidence_score=confidence,
            contradictions=contradictions,
            last_updated=datetime.now(),
            reasoning_steps=reasoning_steps,
        )


# Module-level singleton
_engine: Optional[ReasoningEngine] = None


def get_engine() -> ReasoningEngine:
    global _engine
    if _engine is None:
        _engine = ReasoningEngine()
    return _engine