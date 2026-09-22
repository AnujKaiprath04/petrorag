"""
PetroRAG LLM Provider Abstraction Layer (Module 2.15)
Vendor-neutral LLM client adapters supporting OpenAI, Anthropic, Google Gemini,
Ollama, local vLLM/HuggingFace servers, and an offline Mock provider for
deterministic research benchmarking and CI test execution.
"""

import asyncio
import concurrent.futures
import json
import re
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx

from src.core.config import settings
from src.core.exceptions import LLMGenerationError
from src.core.interfaces import (
    BaseLLMProvider,
    GenerationConfig,
    LLMResponse,
    PromptBundle,
)
from src.core.logging import logger


def _estimate_tokens(text: str) -> int:
    """Accurate token estimator (approx 1 token per 0.75 words / 4 chars)."""
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return max(int(words * 1.33), int(chars / 4.0))


class MockLLMProvider(BaseLLMProvider):
    """
    Offline Mock LLM provider for deterministic test runs, regression testing,
    and offline research benchmarking.
    """

    def __init__(
        self,
        default_response: Optional[str] = None,
        model_name: str = "mock-petrorag-llm"
    ):
        self.model_name = model_name
        self.default_response = default_response or (
            "Based on the provided operational documents, the system parameters "
            "and maintenance thresholds are verified."
        )
        self.response_registry: Dict[str, str] = {}
        self.call_history: List[Dict[str, Any]] = []

    def register_response(self, query_keyword: str, response: str) -> None:
        """Register deterministic canned response for a specific keyword in query."""
        self.response_registry[query_keyword.lower()] = response

    async def generate(
        self,
        prompt: PromptBundle,
        config: Optional[GenerationConfig] = None
    ) -> LLMResponse:
        """Generate response deterministically from registry or prompt context."""
        cfg = config or GenerationConfig()
        query_text = prompt.user_prompt.lower()

        # Find matching canned response if registered
        selected_content = self.default_response
        for kw, canned in self.response_registry.items():
            if kw in query_text:
                selected_content = canned
                break

        # Simulate token usage
        prompt_tokens = _estimate_tokens(prompt.system_prompt + "\n" + prompt.user_prompt)
        completion_tokens = min(_estimate_tokens(selected_content), cfg.max_tokens)
        total_tokens = prompt_tokens + completion_tokens

        resp = LLMResponse(
            content=selected_content,
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason="stop",
            metadata={"provider": "mock", "temperature": cfg.temperature}
        )

        self.call_history.append({
            "prompt": prompt,
            "config": cfg,
            "response": resp
        })
        return resp

    async def generate_stream(
        self,
        prompt: PromptBundle,
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:
        """Asynchronously stream generation tokens word by word."""
        full_response = await self.generate(prompt, config)
        words = full_response.content.split(" ")
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            yield token
            await asyncio.sleep(0.005)

    def generate_sync(
        self,
        prompt: PromptBundle,
        config: Optional[GenerationConfig] = None
    ) -> LLMResponse:
        """Synchronous wrapper for offline testing."""
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, self.generate(prompt, config)).result()


class OpenAILLMProvider(BaseLLMProvider):
    """
    OpenAI-compatible LLM provider adapter.
    Supports official OpenAI models as well as self-hosted OpenAI-compatible
    servers (vLLM, Ollama, LM Studio, TGI).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY or "dummy-key"
        self.model_name = model_name or settings.LLM_MODEL
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _build_messages(self, prompt: PromptBundle) -> List[Dict[str, str]]:
        if prompt.raw_messages:
            return prompt.raw_messages
        return [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.user_prompt}
        ]

    async def generate(
        self,
        prompt: PromptBundle,
        config: Optional[GenerationConfig] = None
    ) -> LLMResponse:
        cfg = config or GenerationConfig()
        messages = self._build_messages(prompt)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "top_p": cfg.top_p or 1.0,
            "stream": False
        }
        if cfg.stop_sequences:
            payload["stop"] = cfg.stop_sequences
        if cfg.seed is not None:
            payload["seed"] = cfg.seed

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                if res.status_code != 200:
                    raise LLMGenerationError(
                        f"OpenAI API error ({res.status_code}): {res.text}",
                        error_code=f"OPENAI_HTTP_{res.status_code}"
                    )
                data = res.json()
                choice = data["choices"][0]
                usage = data.get("usage", {})

                return LLMResponse(
                    content=choice["message"]["content"],
                    model_name=data.get("model", self.model_name),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    finish_reason=choice.get("finish_reason", "stop"),
                    metadata={"provider": "openai", "raw_id": data.get("id")}
                )
        except httpx.RequestError as e:
            raise LLMGenerationError(f"OpenAI connection error: {str(e)}", error_code="OPENAI_NETWORK_ERROR")

    async def generate_stream(
        self,
        prompt: PromptBundle,
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:
        cfg = config or GenerationConfig()
        messages = self._build_messages(prompt)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "stream": True
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status_code != 200:
                        err_body = await response.aread()
                        raise LLMGenerationError(f"OpenAI stream error ({response.status_code}): {err_body.decode('utf-8')}")

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        line_data = line[6:].strip()
                        if line_data == "[DONE]":
                            break
                        try:
                            parsed = json.loads(line_data)
                            delta = parsed["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except Exception:
                            continue
        except httpx.RequestError as e:
            raise LLMGenerationError(f"OpenAI stream connection failed: {str(e)}")


class AnthropicLLMProvider(BaseLLMProvider):
    """
    Anthropic Claude provider adapter.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 60.0
    ):
        self.api_key = api_key or settings.ANTHROPIC_API_KEY or "dummy-anthropic-key"
        self.model_name = model_name or "claude-3-5-sonnet-20241022"
        self.timeout = timeout
        self.endpoint = "https://api.anthropic.com/v1/messages"

    async def generate(
        self,
        prompt: PromptBundle,
        config: Optional[GenerationConfig] = None
    ) -> LLMResponse:
        cfg = config or GenerationConfig()
        payload = {
            "model": self.model_name,
            "system": prompt.system_prompt,
            "messages": [{"role": "user", "content": prompt.user_prompt}],
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(self.endpoint, headers=headers, json=payload)
                if res.status_code != 200:
                    raise LLMGenerationError(f"Anthropic error ({res.status_code}): {res.text}")
                data = res.json()
                content = "".join([blk.get("text", "") for blk in data.get("content", [])])
                usage = data.get("usage", {})
                p_tok = usage.get("input_tokens", 0)
                c_tok = usage.get("output_tokens", 0)

                return LLMResponse(
                    content=content,
                    model_name=data.get("model", self.model_name),
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=p_tok + c_tok,
                    finish_reason=data.get("stop_reason", "stop"),
                    metadata={"provider": "anthropic"}
                )
        except httpx.RequestError as e:
            raise LLMGenerationError(f"Anthropic network error: {str(e)}")

    async def generate_stream(
        self,
        prompt: PromptBundle,
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:
        # Basic non-streaming fallback stream
        resp = await self.generate(prompt, config)
        words = resp.content.split(" ")
        for i, w in enumerate(words):
            yield w + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.005)


class GeminiLLMProvider(BaseLLMProvider):
    """
    Google Gemini provider adapter using REST API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 60.0
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY or "dummy-gemini-key"
        self.model_name = model_name or "gemini-1.5-flash"
        self.timeout = timeout

    async def generate(
        self,
        prompt: PromptBundle,
        config: Optional[GenerationConfig] = None
    ) -> LLMResponse:
        cfg = config or GenerationConfig()
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
            f"?key={self.api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": prompt.system_prompt}]},
            "contents": [{"parts": [{"text": prompt.user_prompt}]}],
            "generationConfig": {
                "temperature": cfg.temperature,
                "maxOutputTokens": cfg.max_tokens,
            }
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(url, json=payload)
                if res.status_code != 200:
                    raise LLMGenerationError(f"Gemini error ({res.status_code}): {res.text}")
                data = res.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise LLMGenerationError("Gemini returned no candidates.")
                text = candidates[0]["content"]["parts"][0]["text"]
                meta = data.get("usageMetadata", {})
                p_tok = meta.get("promptTokenCount", _estimate_tokens(prompt.system_prompt + prompt.user_prompt))
                c_tok = meta.get("candidatesTokenCount", _estimate_tokens(text))

                return LLMResponse(
                    content=text,
                    model_name=self.model_name,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=p_tok + c_tok,
                    finish_reason=candidates[0].get("finishReason", "stop"),
                    metadata={"provider": "gemini"}
                )
        except httpx.RequestError as e:
            raise LLMGenerationError(f"Gemini network failure: {str(e)}")

    async def generate_stream(
        self,
        prompt: PromptBundle,
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:
        resp = await self.generate(prompt, config)
        words = resp.content.split(" ")
        for i, w in enumerate(words):
            yield w + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.005)


class OllamaLLMProvider(BaseLLMProvider):
    """
    Ollama local LLM provider adapter.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 120.0
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model_name = model_name or "llama3.2"
        self.timeout = timeout

    async def generate(
        self,
        prompt: PromptBundle,
        config: Optional[GenerationConfig] = None
    ) -> LLMResponse:
        cfg = config or GenerationConfig()
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt}
            ],
            "options": {
                "temperature": cfg.temperature,
                "num_predict": cfg.max_tokens,
            },
            "stream": False
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(url, json=payload)
                if res.status_code != 200:
                    raise LLMGenerationError(f"Ollama error ({res.status_code}): {res.text}")
                data = res.json()
                content = data.get("message", {}).get("content", "")
                p_tok = data.get("prompt_eval_count", _estimate_tokens(prompt.system_prompt + prompt.user_prompt))
                c_tok = data.get("eval_count", _estimate_tokens(content))

                return LLMResponse(
                    content=content,
                    model_name=self.model_name,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=p_tok + c_tok,
                    finish_reason="stop",
                    metadata={"provider": "ollama"}
                )
        except httpx.RequestError as e:
            raise LLMGenerationError(f"Ollama connection failure: {str(e)}")

    async def generate_stream(
        self,
        prompt: PromptBundle,
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:
        cfg = config or GenerationConfig()
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt}
            ],
            "stream": True
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        raise LLMGenerationError(f"Ollama stream error: {response.status_code}")
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        parsed = json.loads(line)
                        chunk = parsed.get("message", {}).get("content", "")
                        if chunk:
                            yield chunk
                        if parsed.get("done", False):
                            break
        except httpx.RequestError as e:
            raise LLMGenerationError(f"Ollama stream network failure: {str(e)}")


def get_llm_provider(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> BaseLLMProvider:
    """
    Factory function instantiating the requested LLM provider.
    Defaults to settings.LLM_PROVIDER.
    """
    prov = (provider_name or settings.LLM_PROVIDER).lower().strip()

    if prov in ["mock", "test", "offline"]:
        return MockLLMProvider(model_name=model_name or "mock-petrorag-llm")
    elif prov == "openai":
        return OpenAILLMProvider(api_key=api_key, model_name=model_name, base_url=base_url or "https://api.openai.com/v1")
    elif prov in ["anthropic", "claude"]:
        return AnthropicLLMProvider(api_key=api_key, model_name=model_name)
    elif prov in ["gemini", "google"]:
        return GeminiLLMProvider(api_key=api_key, model_name=model_name)
    elif prov == "ollama":
        return OllamaLLMProvider(base_url=base_url, model_name=model_name)
    elif prov in ["local", "vllm"]:
        return OpenAILLMProvider(api_key=api_key or "empty", model_name=model_name or "local-model", base_url=base_url or "http://localhost:8000/v1")
    else:
        logger.warning(f"Unknown provider '{prov}', defaulting to MockLLMProvider.")
        return MockLLMProvider(model_name=model_name or "mock-petrorag-llm")
