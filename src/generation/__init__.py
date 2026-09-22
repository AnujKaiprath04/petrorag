"""
PetroRAG Generation Package
Exports context selection, context compression, prompt construction, and LLM generation adapters.
"""

from src.generation.context_selector import ContextSelector, estimate_tokens
from src.generation.context_compressor import ContextCompressor, split_sentences, extract_parameters
from src.generation.llm_provider import (
    BaseLLMProvider,
    MockLLMProvider,
    OpenAILLMProvider,
    AnthropicLLMProvider,
    GeminiLLMProvider,
    OllamaLLMProvider,
    get_llm_provider,
)
from src.generation.prompt_engine import PromptEngine, DEFAULT_SYSTEM_PROMPT
from src.generation.generator import LLMGenerationService, GenerationResult, GenerationChunk

__all__ = [
    "ContextSelector",
    "ContextCompressor",
    "estimate_tokens",
    "split_sentences",
    "extract_parameters",
    "BaseLLMProvider",
    "MockLLMProvider",
    "OpenAILLMProvider",
    "AnthropicLLMProvider",
    "GeminiLLMProvider",
    "OllamaLLMProvider",
    "get_llm_provider",
    "PromptEngine",
    "DEFAULT_SYSTEM_PROMPT",
    "LLMGenerationService",
    "GenerationResult",
    "GenerationChunk",
]
