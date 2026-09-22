"""
PetroRAG Decision Support REST Endpoints (Module 2.30)
Provides high-performance FastAPI endpoints for technical question answering,
conversational multi-turn dialogues, streaming completions, and session state.
"""

import json
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from backend.app.schemas.rag_response import PetroRAGRequest, PetroRAGResponse
from src.conversation import ConversationManager
from src.core.logging import logger
from src.pipeline import PetroRAGPipeline

router = APIRouter(prefix="/rag", tags=["Decision Support RAG"])

# Global pipeline and conversation manager singletons
_pipeline: Optional[PetroRAGPipeline] = None
_conversation_manager: Optional[ConversationManager] = None


def get_pipeline() -> PetroRAGPipeline:
    """Dependency provider for the unified RAG pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = PetroRAGPipeline()
    return _pipeline


def get_conversation_manager() -> ConversationManager:
    """Dependency provider for dialogue session management."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager


@router.post(
    "/query",
    response_model=PetroRAGResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit technical QA query",
    description="Executes end-to-end retrieval, reranking, LLM generation, grounding verification, and citation extraction."
)
async def query_rag(
    request: PetroRAGRequest,
    session_id: Optional[str] = Query(None, description="Dialogue session ID for context carryover")
) -> PetroRAGResponse:
    pipeline = get_pipeline()
    conv_mgr = get_conversation_manager()

    active_query = request.query
    chat_history = request.chat_history

    # If session_id provided, resolve pronoun references and attach history
    if session_id:
        resolved_q, _ = conv_mgr.resolve_followup_query(session_id, request.query)
        active_query = resolved_q
        if chat_history is None:
            chat_history = conv_mgr.get_chat_history_for_prompt(session_id)

    # Build updated request with resolved query and chat history
    req_payload = PetroRAGRequest(
        query=active_query,
        top_k=request.top_k,
        enable_reranking=request.enable_reranking,
        enable_compression=request.enable_compression,
        enable_grounding=request.enable_grounding,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        chat_history=chat_history,
        metadata_filters=request.metadata_filters
    )

    try:
        response = await pipeline.query(req_payload)

        # If session_id is active, record turn
        if session_id:
            conv_mgr.record_turn(
                session_id=session_id,
                user_query=request.query,
                assistant_answer=response.answer,
                resolved_query=active_query,
                entities=response.entities
            )

        return response
    except Exception as e:
        logger.error(f"Error in RAG endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PetroRAG processing failure: {str(e)}"
        )


@router.post(
    "/stream",
    summary="Stream technical answer tokens",
    description="Streams LLM generation tokens via Server-Sent Events (SSE)."
)
async def stream_rag(
    request: PetroRAGRequest,
    session_id: Optional[str] = Query(None)
):
    pipeline = get_pipeline()
    conv_mgr = get_conversation_manager()

    active_query = request.query
    chat_history = request.chat_history
    if session_id:
        resolved_q, _ = conv_mgr.resolve_followup_query(session_id, request.query)
        active_query = resolved_q
        if chat_history is None:
            chat_history = conv_mgr.get_chat_history_for_prompt(session_id)

    async def event_generator():
        # First retrieve and build context
        val_res = pipeline.validator.validate(active_query)
        clean_q = val_res.sanitized_query
        intent = pipeline.intent_classifier.classify(clean_q).primary_intent
        raw_candidates = pipeline.retriever.retrieve(clean_q, top_k=request.top_k * 3)
        dedup_res = pipeline.deduplicator.deduplicate(raw_candidates)
        deduped = dedup_res.deduplicated_chunks if hasattr(dedup_res, "deduplicated_chunks") else dedup_res
        reranked = pipeline.reranker.rerank(clean_q, deduped, top_k=request.top_k)
        built_context = pipeline.context_selector.build_context(reranked)

        if request.enable_compression:
            built_context = pipeline.context_compressor.compress_built_context(built_context, clean_q)

        async for chunk in pipeline.generator.generate_answer_stream(
            query=clean_q,
            context=built_context,
            intent=intent,
            chat_history=chat_history
        ):
            payload = {
                "delta": chunk.delta,
                "is_final": chunk.is_final,
                "ttft_ms": chunk.time_to_first_token_ms
            }
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get(
    "/sessions/{session_id}",
    summary="Retrieve session state",
    description="Returns dialogue turn history and accumulated equipment context."
)
async def get_session(session_id: str):
    conv_mgr = get_conversation_manager()
    session = conv_mgr.get_or_create_session(session_id)
    return session.model_dump()


@router.delete(
    "/sessions/{session_id}",
    summary="Clear dialogue session",
    description="Deletes active conversation state and accumulated entities."
)
async def clear_session(session_id: str):
    conv_mgr = get_conversation_manager()
    if session_id in conv_mgr.sessions:
        del conv_mgr.sessions[session_id]
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}
