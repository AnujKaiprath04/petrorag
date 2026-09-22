"""
Unit Test: Module 2.17 LLM Generation Service Verification
Verifies:
1. End-to-end prompt composition and completion execution.
2. Latency, TTFT, and throughput (tokens/sec) telemetry.
3. Synchronous execution wrapper for offline evaluation.
4. Asynchronous streaming generator and final boundary chunk.
5. Integration with intent classification and built context.
"""

import pytest
from src.core.interfaces import BuiltContext, GenerationConfig, RerankedChunk
from src.generation.generator import GenerationChunk, GenerationResult, LLMGenerationService
from src.generation.llm_provider import MockLLMProvider
from src.generation.prompt_engine import PromptEngine
from src.query.intent import QueryIntent


def _build_test_context() -> BuiltContext:
    chunk = RerankedChunk(
        chunk_id="chk-comp-01",
        document_id="DOC-C101",
        document_title="Compressor Manual.pdf",
        text="Compressor C-101 trip limit is 7.1 mm/s per ISO 10816-3.",
        initial_score=0.92,
        reranker_score=0.97,
        rank=1,
        page_number=12
    )
    return BuiltContext(
        context_text=f"[DOCUMENT: {chunk.document_title} | Page 12 | ChunkID: {chunk.chunk_id}]\n{chunk.text}",
        chunks=[chunk],
        token_count=25,
        total_candidates_evaluated=1,
        pruned_chunks_count=0,
        preserved_parameters=["7.1 mm/s", "ISO 10816-3"]
    )


@pytest.mark.asyncio
async def test_generation_service_execution_and_telemetry():
    """Verify answer generation with latency, token accounting, and throughput."""
    mock_provider = MockLLMProvider(
        default_response="**Direct Answer**: Centrifugal compressor C-101 vibration trip setpoint is 7.1 mm/s.",
        model_name="mock-petrogpt-v1"
    )
    service = LLMGenerationService(provider=mock_provider, prompt_builder=PromptEngine())

    context = _build_test_context()
    query = "What is the vibration trip limit for compressor C-101?"

    result = await service.generate_answer(
        query=query,
        context=context,
        intent=QueryIntent.TROUBLESHOOTING,
        config=GenerationConfig(temperature=0.0, max_tokens=150)
    )

    assert isinstance(result, GenerationResult)
    assert "7.1 mm/s" in result.content
    assert result.model_name == "mock-petrogpt-v1"
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0
    assert result.total_tokens == result.prompt_tokens + result.completion_tokens
    assert result.generation_time_ms >= 0.0
    assert result.tokens_per_second >= 0.0
    assert "<context>" in result.prompt_bundle.user_prompt


def test_generation_service_sync_wrapper():
    """Verify synchronous execution wrapper for benchmarking scripts."""
    mock_provider = MockLLMProvider(
        default_response="Synchronously generated response."
    )
    service = LLMGenerationService(provider=mock_provider)
    context = _build_test_context()

    result = service.generate_answer_sync(
        query="Quick sync question",
        context=context
    )

    assert isinstance(result, GenerationResult)
    assert result.content == "Synchronously generated response."
    assert result.total_tokens > 0


@pytest.mark.asyncio
async def test_generation_service_streaming_with_ttft():
    """Verify token streaming with TTFT (Time to First Token) telemetry."""
    mock_provider = MockLLMProvider(
        default_response="Emergency shutdown occurs at 7.1 mm/s."
    )
    service = LLMGenerationService(provider=mock_provider)
    context = _build_test_context()

    chunks = []
    async for chunk in service.generate_answer_stream(
        query="What is the trip setpoint?",
        context=context
    ):
        chunks.append(chunk)

    assert len(chunks) > 1
    # Check that initial chunk has TTFT recorded
    first_chunk = chunks[0]
    assert isinstance(first_chunk, GenerationChunk)
    assert first_chunk.time_to_first_token_ms is not None
    assert first_chunk.time_to_first_token_ms >= 0.0

    # Check final boundary chunk
    final_chunk = chunks[-1]
    assert final_chunk.is_final is True
    assert final_chunk.delta == ""
    assert final_chunk.accumulated == "Emergency shutdown occurs at 7.1 mm/s."
