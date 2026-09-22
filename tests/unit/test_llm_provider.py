"""
Unit Test: Module 2.15 LLM Provider Abstraction Verification
Verifies:
1. MockLLMProvider async generation, token accounting, and metadata.
2. Synchronous wrapper generation (generate_sync).
3. Asynchronous streaming generation.
4. Custom canned response registration and matching.
5. Provider factory instantiation (OpenAI, Anthropic, Gemini, Ollama, Local, Mock).
6. Exception hierarchy handling for network/API failures.
"""

import pytest
from src.core.exceptions import LLMGenerationError
from src.core.interfaces import (
    GenerationConfig,
    LLMResponse,
    PromptBundle,
)
from src.generation.llm_provider import (
    AnthropicLLMProvider,
    GeminiLLMProvider,
    MockLLMProvider,
    OllamaLLMProvider,
    OpenAILLMProvider,
    get_llm_provider,
)


@pytest.mark.asyncio
async def test_mock_llm_provider_async_generation():
    """Verify asynchronous generation and token telemetry."""
    provider = MockLLMProvider(
        default_response="Centrifugal compressor C-101 vibration trip limit is 7.1 mm/s per ISO 10816-3.",
        model_name="mock-petro-gpt"
    )

    prompt = PromptBundle(
        system_prompt="You are an Oil & Gas expert. Answer strictly from context.",
        user_prompt="What is the vibration trip limit for compressor C-101?"
    )
    config = GenerationConfig(temperature=0.0, max_tokens=256)

    response = await provider.generate(prompt, config)

    assert isinstance(response, LLMResponse)
    assert "7.1 mm/s" in response.content
    assert response.model_name == "mock-petro-gpt"
    assert response.prompt_tokens > 0
    assert response.completion_tokens > 0
    assert response.total_tokens == response.prompt_tokens + response.completion_tokens
    assert response.finish_reason == "stop"
    assert response.metadata["provider"] == "mock"


def test_mock_llm_provider_sync_wrapper():
    """Verify synchronous execution wrapper."""
    provider = MockLLMProvider(default_response="Synchronous test response.")
    prompt = PromptBundle(
        system_prompt="Test system prompt.",
        user_prompt="Test user query."
    )
    resp = provider.generate_sync(prompt)
    assert resp.content == "Synchronous test response."
    assert resp.total_tokens > 0


@pytest.mark.asyncio
async def test_mock_llm_provider_streaming():
    """Verify asynchronous streaming yields sequential text tokens."""
    provider = MockLLMProvider(
        default_response="Normal separator pressure is 45.0 bar."
    )
    prompt = PromptBundle(
        system_prompt="System instructions.",
        user_prompt="Query about separator pressure."
    )

    chunks = []
    async for chunk in provider.generate_stream(prompt):
        chunks.append(chunk)

    reconstructed = "".join(chunks)
    assert reconstructed == "Normal separator pressure is 45.0 bar."
    assert len(chunks) > 1  # Streamed token by token / word by word


@pytest.mark.asyncio
async def test_mock_llm_provider_canned_registry():
    """Verify keyword-based canned responses for deterministic testing."""
    provider = MockLLMProvider(default_response="Generic fallback response.")
    provider.register_response(
        query_keyword="vibration",
        response="High vibration alarm triggers at 4.5 mm/s."
    )
    provider.register_response(
        query_keyword="flare",
        response="Emergency depressurization routed to high pressure flare system."
    )

    p1 = PromptBundle(system_prompt="", user_prompt="Why is vibration high on C-101?")
    r1 = await provider.generate(p1)
    assert "4.5 mm/s" in r1.content

    p2 = PromptBundle(system_prompt="", user_prompt="Where does flare vent go?")
    r2 = await provider.generate(p2)
    assert "high pressure flare system" in r2.content

    p3 = PromptBundle(system_prompt="", user_prompt="Who is the operator?")
    r3 = await provider.generate(p3)
    assert r3.content == "Generic fallback response."


def test_get_llm_provider_factory():
    """Verify factory instantiation across all supported provider variants."""
    p_mock = get_llm_provider("mock")
    assert isinstance(p_mock, MockLLMProvider)

    p_openai = get_llm_provider("openai", api_key="sk-test", model_name="gpt-4o")
    assert isinstance(p_openai, OpenAILLMProvider)
    assert p_openai.model_name == "gpt-4o"

    p_anthropic = get_llm_provider("anthropic", api_key="ant-test", model_name="claude-3-sonnet")
    assert isinstance(p_anthropic, AnthropicLLMProvider)

    p_gemini = get_llm_provider("gemini", api_key="gem-test", model_name="gemini-1.5-pro")
    assert isinstance(p_gemini, GeminiLLMProvider)

    p_ollama = get_llm_provider("ollama", base_url="http://localhost:11434")
    assert isinstance(p_ollama, OllamaLLMProvider)

    p_local = get_llm_provider("local", base_url="http://localhost:8000/v1")
    assert isinstance(p_local, OpenAILLMProvider)

    # Unknown defaults to MockLLMProvider gracefully
    p_unknown = get_llm_provider("unsupported_vendor")
    assert isinstance(p_unknown, MockLLMProvider)


@pytest.mark.asyncio
async def test_openai_provider_error_handling():
    """Verify that network failures raise domain-specific LLMGenerationError."""
    # Point to a dead/unreachable port
    provider = OpenAILLMProvider(
        api_key="test-key",
        base_url="http://127.0.0.1:9999/v1",
        timeout=0.5
    )
    prompt = PromptBundle(
        system_prompt="Test system.",
        user_prompt="Test query."
    )

    with pytest.raises(LLMGenerationError) as exc_info:
        await provider.generate(prompt)

    assert "OPENAI" in exc_info.value.error_code or "connection" in exc_info.value.message.lower()
