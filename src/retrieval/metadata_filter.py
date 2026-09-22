"""
PetroRAG Metadata-Aware Retrieval Engine (Module 2.9)
Extracts, validates, and applies domain metadata filters (field, well, asset,
equipment, equipment_type, document_type, department, date, revision)
with adaptive relaxation to prevent accidental exclusion of critical evidence.
"""

from typing import Any, Dict, List, Optional
from src.core.interfaces import BaseRetriever, RetrievedChunk
from src.core.logging import logger
from src.query.intent import QueryIntent, HybridIntentClassifier
from src.query.entities import QueryEntities, RuleBasedEntityExtractor


# Supported canonical metadata keys
SUPPORTED_METADATA_KEYS = {
    "field",
    "well",
    "asset",
    "equipment",
    "equipment_id",
    "equipment_type",
    "document_type",
    "department",
    "date",
    "revision",
}

# Intent to document_type heuristics
INTENT_TO_DOC_TYPE = {
    QueryIntent.MAINTENANCE: "maintenance",
    QueryIntent.TROUBLESHOOTING: "troubleshooting",
    QueryIntent.SAFETY: "safety",
    QueryIntent.DOCUMENT_SEARCH: "sop",
    QueryIntent.INCIDENT: "incident_report",
}


class MetadataFilterEngine:
    """
    Constructs and applies metadata constraints to retrieval pipelines.
    Implements adaptive relaxation: if strict filtering yields insufficient candidates,
    it progressively relaxes secondary constraints to ensure critical evidence is never lost.
    """

    def __init__(
        self,
        entity_extractor: Optional[RuleBasedEntityExtractor] = None,
        intent_classifier: Optional[HybridIntentClassifier] = None
    ):
        self.entity_extractor = entity_extractor or RuleBasedEntityExtractor()
        self.intent_classifier = intent_classifier or HybridIntentClassifier()

    def build_filter(
        self,
        query: str,
        entities: Optional[QueryEntities] = None,
        intent: Optional[QueryIntent] = None
    ) -> Dict[str, Any]:
        """
        Derive metadata filter dictionary from query syntax, extracted entities, and intent.
        Example:
        Query: What is the maintenance procedure for compressor C-101?
        Filter: equipment = C-101, document_type = maintenance
        """
        if not entities:
            entities = self.entity_extractor.extract(query)
        if not intent:
            intent = self.intent_classifier.classify(query).primary_intent

        filters: Dict[str, Any] = {}

        # 1. Equipment & Equipment Type
        if entities.equipment_ids:
            filters["equipment_id"] = entities.equipment_ids[0]
            # Standardize alias 'equipment'
            filters["equipment"] = entities.equipment_ids[0]
        if entities.equipment_types:
            filters["equipment_type"] = entities.equipment_types[0]

        # 2. Field, Well, Asset
        if entities.fields:
            filters["field"] = entities.fields[0]
        if entities.wells:
            filters["well"] = entities.wells[0]
        if entities.assets:
            filters["asset"] = entities.assets[0]

        # 3. Document Type mapped from intent or keywords
        q_lower = query.lower()
        if "maintenance" in q_lower or intent == QueryIntent.MAINTENANCE:
            filters["document_type"] = "maintenance"
        elif "safety" in q_lower or "loto" in q_lower or intent == QueryIntent.SAFETY:
            filters["document_type"] = "safety"
        elif "incident" in q_lower or intent == QueryIntent.INCIDENT:
            filters["document_type"] = "incident_report"
        elif "manual" in q_lower:
            filters["document_type"] = "manual"
        elif "datasheet" in q_lower:
            filters["document_type"] = "datasheet"

        # 4. Dates / Revisions
        if entities.dates:
            d_val = entities.dates[0]
            if "rev" in d_val.lower():
                filters["revision"] = d_val
            else:
                filters["date"] = d_val

        return filters

    def matches_filter(
        self,
        chunk: RetrievedChunk,
        filters: Dict[str, Any],
        strict: bool = True
    ) -> bool:
        """
        Evaluates whether a chunk satisfies the metadata filter constraints.
        If strict=False, missing metadata fields in the chunk do not cause disqualification.
        """
        if not filters:
            return True

        meta = chunk.metadata or {}

        for key, expected in filters.items():
            if expected is None:
                continue

            # Check direct attribute or metadata dictionary
            actual = meta.get(key)
            if actual is None:
                # Check normalized aliases
                if key == "equipment":
                    actual = meta.get("equipment_id") or meta.get("tag")
                elif key == "equipment_id":
                    actual = meta.get("equipment") or meta.get("tag")

            if actual is None:
                if strict:
                    return False
                continue

            if str(actual).lower() != str(expected).lower():
                return False

        return True

    def execute_metadata_aware_retrieval(
        self,
        query: str,
        retriever: BaseRetriever,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        min_candidates: int = 1
    ) -> List[RetrievedChunk]:
        """
        Two-pass adaptive metadata retrieval:
        1. Pass 1 (Strict Filter): High-precision candidate generation.
        2. Pass 2 (Adaptive Relaxation): If strict filtering returns fewer than min_candidates (e.g. 0),
           progressively relaxes non-primary filters (e.g. document_type or department)
           to ensure relevant evidence is not lost due to missing or imperfect metadata.
        """
        active_filters = filters if filters is not None else self.build_filter(query)

        # Pass 1: Strict filtering
        results = retriever.retrieve(query, top_k=top_k, filters=active_filters)
        if len(results) >= min_candidates:
            return results

        logger.info(
            f"Strict metadata filter {active_filters} yielded {len(results)} (< {min_candidates}) candidates. "
            "Applying adaptive metadata relaxation."
        )

        # Pass 2: Progressive Relaxation
        # Relax secondary filter: document_type or department, preserving equipment_id
        relaxed_filters = dict(active_filters)
        for relax_key in ["document_type", "department", "date", "revision"]:
            if relax_key in relaxed_filters:
                del relaxed_filters[relax_key]

        if relaxed_filters != active_filters:
            relaxed_results = retriever.retrieve(query, top_k=top_k, filters=relaxed_filters)
            if len(relaxed_results) >= min_candidates:
                return relaxed_results

        # Pass 3: Unfiltered fallback with soft metadata re-scoring
        unfiltered_results = retriever.retrieve(query, top_k=top_k * 2, filters=None)

        # Re-score: boost chunks matching any filter attributes, keep others
        scored_results: List[RetrievedChunk] = []
        for chunk in unfiltered_results:
            boost = 1.0
            for k, v in active_filters.items():
                meta_val = chunk.metadata.get(k)
                if meta_val and str(meta_val).lower() == str(v).lower():
                    boost += 0.25
            chunk.score = round(chunk.score * boost, 4)
            scored_results.append(chunk)

        scored_results.sort(key=lambda c: c.score, reverse=True)
        return scored_results[:top_k]
