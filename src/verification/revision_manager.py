"""
PetroRAG Document Revision Awareness & Recency Weighting Engine (Module 2.27)
Identifies document revisions, detects superseded operational manuals,
and resolves version conflicts in favor of latest approved standards.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from src.core.interfaces import RerankedChunk
from src.core.logging import logger


class RevisionMetadata(BaseModel):
    """Extracted revision details for a document chunk."""
    revision_number: float = Field(default=1.0, description="Parsed numerical revision")
    revision_string: str = Field(default="Rev 1")
    year: Optional[int] = None
    is_superseded: bool = Field(default=False)
    superseded_by_revision: Optional[str] = None


class RevisionManager:
    """
    Parses document versioning, orders document families chronologically,
    and boosts latest approved revisions while flagging obsolete documents.
    """

    REVISION_REGEX = re.compile(
        r"\b(?:Rev(?:ision)?\.?|Ed(?:ition)?\.?|v(?:ersion)?\.?)\s*(\d+(?:\.\d+)?)\b",
        re.IGNORECASE
    )
    YEAR_REGEX = re.compile(r"\b(19\d{2}|20\d{2})\b")

    def extract_revision(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> RevisionMetadata:
        """
        Extracts revision number and publication year from text and metadata.
        """
        meta = metadata or {}
        combined = f"{meta.get('document_title', '')} {meta.get('document_id', '')} {text[:300]}"

        rev_match = self.REVISION_REGEX.search(combined)
        year_match = self.YEAR_REGEX.search(combined)

        rev_num = float(rev_match.group(1)) if rev_match else 1.0
        rev_str = rev_match.group(0) if rev_match else f"Rev {int(rev_num)}"
        year = int(year_match.group(1)) if year_match else None

        return RevisionMetadata(
            revision_number=rev_num,
            revision_string=rev_str,
            year=year
        )

    def resolve_revisions(
        self,
        chunks: List[RerankedChunk],
        boost_multiplier: float = 1.25,
        penalize_superseded: bool = True
    ) -> List[RerankedChunk]:
        """
        Groups chunks by document family (based on title or equipment),
        identifies the latest revision, boosts its reranker_score, and marks
        superseded chunks.
        """
        if len(chunks) <= 1:
            return chunks

        # Group by normalized document family
        families: Dict[str, List[Tuple[RerankedChunk, RevisionMetadata]]] = {}

        for c in chunks:
            title = (c.document_title or c.document_id or "").lower()
            # Normalize title to base family name (strip Rev, version, year)
            base_family = re.sub(r"\brev(?:ision)?\.?\s*\d+(?:\.\d+)?\b", "", title, flags=re.IGNORECASE)
            base_family = re.sub(r"\b(19\d{2}|20\d{2})\b", "", base_family).strip()
            base_family = re.sub(r"[\-_\.\s]+", " ", base_family).strip()

            rev_meta = self.extract_revision(c.text, c.metadata)
            families.setdefault(base_family, []).append((c, rev_meta))

        updated_chunks: List[RerankedChunk] = []

        for family_name, member_list in families.items():
            if len(member_list) == 1:
                updated_chunks.append(member_list[0][0])
                continue

            # Find latest revision within family
            max_rev = max(item[1].revision_number for item in member_list)
            max_item = max(member_list, key=lambda x: x[1].revision_number)
            latest_str = max_item[1].revision_string

            for chunk, rev_meta in member_list:
                meta = dict(chunk.metadata)
                if rev_meta.revision_number < max_rev:
                    # Superseded chunk
                    rev_meta.is_superseded = True
                    rev_meta.superseded_by_revision = latest_str
                    meta["is_superseded"] = True
                    meta["superseded_by"] = latest_str

                    new_score = chunk.reranker_score * 0.70 if penalize_superseded else chunk.reranker_score
                    logger.info(f"Chunk {chunk.chunk_id} flagged as SUPERSEDED by {latest_str}")
                else:
                    # Latest approved revision
                    meta["is_superseded"] = False
                    meta["is_latest_revision"] = True
                    new_score = min(1.0, chunk.reranker_score * boost_multiplier)

                updated_chunks.append(RerankedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    text=chunk.text,
                    initial_score=chunk.initial_score,
                    reranker_score=round(new_score, 4),
                    rank=chunk.rank,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    metadata=meta
                ))

        # Re-sort by updated score descending and recalculate ranks
        updated_chunks.sort(key=lambda x: x.reranker_score, reverse=True)
        for idx, c in enumerate(updated_chunks):
            c.rank = idx + 1

        return updated_chunks
