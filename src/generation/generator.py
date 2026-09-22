"""
PetroRAG LLM Generation Service (Module 2.17)
High-level generation orchestrator that connects prompt composition,
LLM providers, telemetry tracking (TTFT, latency, tokens/sec),
and streaming generation.
"""

import asyncio
import concurrent.futures
import time
from typing import Any, AsyncIterator, Dict, List, Optional
from pydantic import BaseModel, Field

from src.core.exceptions import LLMGenerationError
from src.core.interfaces import (
    BaseLLMProvider,
    BasePromptBuilder,
    BuiltContext,
    GenerationConfig,
    LLMResponse,
    PromptBundle,
)
from src.core.logging import logger
from src.generation.llm_provider import get_llm_provider
from src.generation.prompt_engine import PromptEngine
from src.query.intent import QueryIntent


class GenerationResult(BaseModel):
    """Telemetry-enriched response from the generation service."""
    content: str = Field(..., description="Generated answer text")
    model_name: str = Field(..., description="Model identifier")
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    generation_time_ms: float = Field(default=0.0, description="Generation duration in milliseconds")
    tokens_per_second: float = Field(default=0.0, description="Generation throughput")
    prompt_bundle: PromptBundle
    finish_reason: Optional[str] = "stop"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GenerationChunk(BaseModel):
    """Streaming chunk payload with latency telemetry."""
    delta: str = Field(..., description="Incremental text token or segment")
    accumulated: str = Field(..., description="Full text accumulated so far")
    is_final: bool = Field(default=False)
    time_to_first_token_ms: Optional[float] = None


class LLMGenerationService:
    """
    Orchestrates prompt construction and LLM completion execution,
    tracking runtime latency, token usage, and streaming events.
    """

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        prompt_builder: Optional[BasePromptBuilder] = None
    ):
        self.provider = provider or get_llm_provider()
        self.prompt_builder = prompt_builder or PromptEngine()

    async def generate_answer(
        self,
        query: str,
        context: BuiltContext,
        intent: Optional[QueryIntent] = None,
        config: Optional[GenerationConfig] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_instruction: Optional[str] = None
    ) -> GenerationResult:
        """
        Builds structured prompt and executes LLM generation with performance telemetry.
        """
        cfg = config or GenerationConfig()

        prompt = self.prompt_builder.build_prompt(
            query=query,
            context=context,
            system_instruction=system_instruction,
            chat_history=chat_history,
            intent=intent
        )

        start_time = time.perf_counter()
        try:
            response: LLMResponse = await self.provider.generate(prompt, config=cfg)
        except Exception as e:
            logger.error(f"Generation failure: {str(e)}")
            if isinstance(e, LLMGenerationError):
                raise
            raise LLMGenerationError(f"Unexpected generation failure: {str(e)}")

        duration = time.perf_counter() - start_time
        duration_ms = round(duration * 1000.0, 2)
        tokens_per_sec = round(response.completion_tokens / duration, 2) if duration > 0 else 0.0

        logger.info(
            f"Generated answer in {duration_ms}ms ({tokens_per_sec} tok/s), "
            f"tokens: {response.prompt_tokens}p + {response.completion_tokens}c = {response.total_tokens}t"
        )

        return GenerationResult(
            content=response.content,
            model_name=response.model_name,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            generation_time_ms=duration_ms,
            tokens_per_second=tokens_per_sec,
            prompt_bundle=prompt,
            finish_reason=response.finish_reason,
            metadata=response.metadata
        )

    async def generate_answer_stream(
        self,
        query: str,
        context: BuiltContext,
        intent: Optional[QueryIntent] = None,
        config: Optional[GenerationConfig] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_instruction: Optional[str] = None
    ) -> AsyncIterator[GenerationChunk]:
        """
        Asynchronously streams completion tokens with TTFT telemetry.
        """
        cfg = config or GenerationConfig(stream=True)

        prompt = self.prompt_builder.build_prompt(
            query=query,
            context=context,
            system_instruction=system_instruction,
            chat_history=chat_history,
            intent=intent
        )

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        accumulated_text = ""

        try:
            async for token in self.provider.generate_stream(prompt, config=cfg):
                if first_token_time is None:
                    first_token_time = round((time.perf_counter() - start_time) * 1000.0, 2)

                accumulated_text += token
                yield GenerationChunk(
                    delta=token,
                    accumulated=accumulated_text,
                    is_final=False,
                    time_to_first_token_ms=first_token_time
                )

            # Final boundary chunk
            yield GenerationChunk(
                delta="",
                accumulated=accumulated_text,
                is_final=True,
                time_to_first_token_ms=first_token_time
            )
        except Exception as e:
            logger.error(f"Streaming failure: {str(e)}")
            if isinstance(e, LLMGenerationError):
                raise
            raise LLMGenerationError(f"Streaming failure: {str(e)}")

    def generate_answer_sync(
        self,
        query: str,
        context: BuiltContext,
        intent: Optional[QueryIntent] = None,
        config: Optional[GenerationConfig] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_instruction: Optional[str] = None
    ) -> GenerationResult:
        """Synchronous wrapper for offline evaluation benchmarks."""
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(
                asyncio.run,
                self.generate_answer(
                    query=query,
                    context=context,
                    intent=intent,
                    config=config,
                    chat_history=chat_history,
                    system_instruction=system_instruction
                )
            ).result()
