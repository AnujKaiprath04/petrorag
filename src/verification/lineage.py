"""
PetroRAG Source Traceability Lineage Engine (Module 2.20)
Provides end-to-end provenance tracking, cryptographic content hashing,
and regulatory compliance auditing for every cited claim.
"""

from datetime import datetime, timezone
import hashlib
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.core.interfaces import BuiltContext, Citation, RerankedChunk
from src.core.logging import logger


class LineageRecord(BaseModel):
    """
    Unbroken audit provenance record linking a cited claim to its physical source.
    """
    lineage_id: str = Field(..., description="Unique lineage tracking ID")
    citation_id: str = Field(..., description="Citation handle in answer")
    document_id: str = Field(..., description="Document unique identifier")
    document_title: Optional[str] = Field(None, description="Human-readable document title")
    page_number: Optional[int] = Field(None, description="1-indexed source page number")
    section_title: Optional[str] = Field(None, description="Section heading")
    chunk_id: str = Field(..., description="Chunk ID")
    chunk_sha256: str = Field(..., description="Cryptographic SHA-256 hash of original chunk text")
    retrieval_channel: str = Field(default="hybrid", description="Retrieval source channel")
    reranker_score: float = Field(default=0.0, description="Cross-encoder relevance score")
    rank: int = Field(default=1, description="Context priority rank")
    verbatim_quote: str = Field(..., description="Exact textual excerpt supporting citation")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LineageManifest(BaseModel):
    """
    Audit-ready manifest containing full provenance for an answer session.
    """
    manifest_id: str = Field(..., description="Manifest identifier")
    answer_snippet: str = Field(..., description="Answer text preview")
    total_citations: int = Field(default=0)
    records: List[LineageRecord] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_tamper_free: bool = Field(default=True)


class SourceTraceabilityAuditor:
    """
    Constructs, audits, and cryptographically verifies lineage provenance manifests.
    """

    @staticmethod
    def _compute_sha256(text: str) -> str:
        """Compute SHA-256 hex digest of text content."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def build_lineage(
        self,
        answer_text: str,
        citations: List[Citation],
        context: BuiltContext
    ) -> LineageManifest:
        """
        Builds a comprehensive LineageManifest mapping all citations to context chunks.
        """
        chunk_map: Dict[str, RerankedChunk] = {c.chunk_id: c for c in context.chunks}
        records: List[LineageRecord] = []

        for idx, cit in enumerate(citations):
            chunk = chunk_map.get(cit.chunk_id)
            chunk_text = chunk.text if chunk else cit.text_snippet
            chunk_hash = self._compute_sha256(chunk_text)
            channel = str(chunk.metadata.get("channel", "hybrid")) if chunk else "hybrid"
            rank = chunk.rank if chunk else (idx + 1)

            record = LineageRecord(
                lineage_id=f"lin-{idx + 1:04d}",
                citation_id=cit.citation_id,
                document_id=cit.document_id,
                document_title=cit.document_title,
                page_number=cit.page_number,
                section_title=cit.section_title,
                chunk_id=cit.chunk_id,
                chunk_sha256=chunk_hash,
                retrieval_channel=channel,
                reranker_score=cit.relevance_score,
                rank=rank,
                verbatim_quote=cit.text_snippet
            )
            records.append(record)

        manifest_id = f"man-{self._compute_sha256(answer_text)[:12]}"
        manifest = LineageManifest(
            manifest_id=manifest_id,
            answer_snippet=answer_text[:120].strip() + ("..." if len(answer_text) > 120 else ""),
            total_citations=len(records),
            records=records,
            is_tamper_free=True
        )

        logger.info(f"Generated LineageManifest {manifest_id} with {len(records)} records.")
        return manifest

    def verify_manifest_integrity(
        self,
        manifest: LineageManifest,
        context: BuiltContext
    ) -> bool:
        """
        Cryptographically verifies that all chunk SHA-256 hashes in manifest
        match the active context chunks.
        """
        chunk_map: Dict[str, RerankedChunk] = {c.chunk_id: c for c in context.chunks}

        for rec in manifest.records:
            target_chunk = chunk_map.get(rec.chunk_id)
            if not target_chunk:
                return False
            expected_hash = self._compute_sha256(target_chunk.text)
            if rec.chunk_sha256 != expected_hash:
                return False

        return True
