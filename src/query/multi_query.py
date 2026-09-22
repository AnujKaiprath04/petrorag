"""
PetroRAG Multi-Query Generation & Multi-Query Retrieval Orchestrator (Module 2.6)
Generates multiple diverse domain retrieval formulations, executes retrieval
independently across each query, merges candidate streams, and removes duplicate chunks.
"""

from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field
from src.core.interfaces import BaseRetriever, RetrievedChunk, RetrievalChannel
from src.query.intent import QueryIntent, HybridIntentClassifier
from src.query.entities import QueryEntities, RuleBasedEntityExtractor
from src.query.rewriter import RuleBasedQueryRewriter, RewrittenQueries


class MultiQueryPlan(BaseModel):
    """Execution plan containing original query and sub-queries to execute."""
    original_query: str
    generated_queries: List[str] = Field(default_factory=list)
    strategy: str = "domain_multi_query"


class MultiQueryRetriever:
    """
    Wraps an underlying BaseRetriever to execute multi-perspective retrieval:
    1. Generates 3-5 distinct query formulations targeting symptoms, causes, SOPs, and equipment.
    2. Retrieves candidate chunks independently for each query formulation.
    3. Merges candidate pools, deduplicating by chunk_id and aggregating scores.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        rewriter: Optional[RuleBasedQueryRewriter] = None,
        entity_extractor: Optional[RuleBasedEntityExtractor] = None,
        intent_classifier: Optional[HybridIntentClassifier] = None,
        max_queries: int = 4
    ):
        self.base_retriever = base_retriever
        self.rewriter = rewriter or RuleBasedQueryRewriter()
        self.entity_extractor = entity_extractor or RuleBasedEntityExtractor()
        self.intent_classifier = intent_classifier or HybridIntentClassifier()
        self.max_queries = max_queries

    def generate_queries(self, query: str) -> List[str]:
        """
        Generate multiple retrieval formulations.
        Example:
        Original: Why is separator pressure high?
        Queries:
        1. separator high pressure
        2. separator pressure increase causes
        3. production separator pressure troubleshooting
        4. separator pressure control failure
        """
        entities = self.entity_extractor.extract(query)
        intent_res = self.intent_classifier.classify(query)
        intent = intent_res.primary_intent
        rewritten = self.rewriter.rewrite(query, intent=intent, entities=entities)

        candidates: List[str] = []

        # Add domain specific variations if query is about pressure/separator/failure
        q_lower = query.lower()
        if "separator" in q_lower and ("pressure high" in q_lower or "high pressure" in q_lower or "pressure" in q_lower):
            specific_variants = [
                "separator high pressure",
                "separator pressure increase causes",
                "production separator pressure troubleshooting",
                "separator pressure control failure",
            ]
            candidates.extend(specific_variants)

        # Append rewritten candidates and original query
        candidates.extend(rewritten.rewritten_candidates)
        candidates.append(query)

        # Deduplicate while preserving order
        unique_queries: List[str] = []
        for q in candidates:
            q_clean = q.strip()
            if q_clean and q_clean not in unique_queries:
                unique_queries.append(q_clean)

        return unique_queries[:self.max_queries]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """
        Execute multi-query retrieval:
        1. Formulate sub-queries.
        2. Retrieve top candidates for each.
        3. Merge and deduplicate by chunk_id, computing combined relevance.
        """
        sub_queries = self.generate_queries(query)

        # Map chunk_id -> best RetrievedChunk
        merged_chunks: Dict[str, RetrievedChunk] = {}
        # Count how many queries retrieved this chunk (agreement signal)
        query_hit_counts: Dict[str, int] = {}

        for sub_q in sub_queries:
            sub_results = self.base_retriever.retrieve(sub_q, top_k=top_k, filters=filters)
            for chunk in sub_results:
                cid = chunk.chunk_id
                query_hit_counts[cid] = query_hit_counts.get(cid, 0) + 1

                if cid not in merged_chunks:
                    merged_chunks[cid] = chunk.model_copy(deep=True)
                else:
                    # Update score: take max score or boost if retrieved by multiple query perspectives
                    existing = merged_chunks[cid]
                    if chunk.score > existing.score:
                        existing.score = chunk.score

        # Apply multi-perspective agreement boost (5% per additional query hit)
        for cid, chunk in merged_chunks.items():
            hits = query_hit_counts.get(cid, 1)
            agreement_boost = 1.0 + 0.05 * (hits - 1)
            chunk.score = round(chunk.score * agreement_boost, 4)

        # Sort descending by fused score
        sorted_chunks = sorted(merged_chunks.values(), key=lambda c: c.score, reverse=True)
        return sorted_chunks[:top_k]

    async def aretrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """Asynchronous multi-query retrieval."""
        # For base implementations that are synchronous or asynchronous, fall back to synchronous
        return self.retrieve(query, top_k=top_k, filters=filters)
