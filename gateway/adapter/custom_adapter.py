"""
Custom Provider Adapter untuk LegionADK.
Mendukung penyedia kustom (custom_provider) secara dinamis via CUSTOM_PROVIDER_BASE_URL.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

# pyrefly: ignore[missing-import]
import httpx

logger = logging.getLogger("LegionADK.gateway.adapter.custom")

from LegionADK.gateway.adapter.base_adapter import BaseLLMAdapter

if TYPE_CHECKING:
    from LegionADK.core.llm_spec import CancellationToken


class CustomAdapter(BaseLLMAdapter):
    """
    Adapter fleksibel untuk penyedia kustom (custom_provider).
    Menerima base_url dan api_key secara dinamis via parameter atau
    variabel lingkungan CUSTOM_PROVIDER_BASE_URL dan CUSTOM_PROVIDER_API_KEY.
    """

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_url = base_url.strip().rstrip("/")
        self.api_key = api_key
        self.default_headers = default_headers or {}

    def resolve_base_url(self, base_url: str = "") -> str:
        """
        Menentukan base URL dari argumen fungsi, atribut instance,
        atau variabel lingkungan CUSTOM_PROVIDER_BASE_URL.
        """
        url = (
            base_url
            or self.base_url
            or os.getenv("CUSTOM_PROVIDER_BASE_URL", "")
        )
        clean_url = url.strip().rstrip("/")
        if not clean_url:
            raise ValueError(
                "base_url tidak boleh kosong. Harap berikan base_url melalui parameter panggilan, "
                "instansiasi CustomAdapter(base_url=...), atau atur variabel lingkungan CUSTOM_PROVIDER_BASE_URL."
            )
        return clean_url

    async def complete(
        self,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: str = "",
        model: str = "",
        api_key: str = "",
        base_url: str = "",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        images: Optional[List[str]] = None,
        thinking_level: str = "medium",
        token: Optional[CancellationToken] = None,
        provider: str = "custom",
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Menjalankan pemanggilan completion teks secara asinkron ke endpoint kustom."""
        if token is not None:
            await token.wait_if_paused()
            token.raise_if_cancelled()

        target_base_url = self.resolve_base_url(base_url=base_url)
        url = f"{target_base_url}/chat/completions"

        # 1. Bangun daftar messages
        formatted_messages: List[Dict[str, Any]] = []

        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        if messages:
            # Salin dictionary per item agar tidak memutasi input messages milik pemanggil
            formatted_messages.extend([dict(m) for m in messages])
            # Lampirkan images ke turn user terakhir jika ada
            if images:
                for idx in range(len(formatted_messages) - 1, -1, -1):
                    msg = formatted_messages[idx]
                    if msg.get("role") == "user":
                        orig_content = msg.get("content", "")
                        content_parts: List[Dict[str, Any]] = []
                        if isinstance(orig_content, str):
                            if orig_content:
                                content_parts.append({"type": "text", "text": orig_content})
                        elif isinstance(orig_content, list):
                            content_parts.extend(orig_content)

                        for img in images:
                            img_url = (
                                img
                                if img.startswith(("http://", "https://", "data:"))
                                else f"data:image/jpeg;base64,{img}"
                            )
                            content_parts.append({"type": "image_url", "image_url": {"url": img_url}})
                        new_msg = dict(msg)
                        new_msg["content"] = content_parts
                        formatted_messages[idx] = new_msg
                        break
        elif prompt:
            user_content: Union[str, List[Dict[str, Any]]] = prompt
            if images:
                content_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
                for img in images:
                    img_url = (
                        img
                        if img.startswith(("http://", "https://", "data:"))
                        else f"data:image/jpeg;base64,{img}"
                    )
                    content_parts.append({"type": "image_url", "image_url": {"url": img_url}})
                user_content = content_parts
            formatted_messages.append({"role": "user", "content": user_content})

        # 2. Siapkan payload request OpenAI-compatible
        payload: Dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if thinking_level and thinking_level in ("low", "medium", "high"):
            payload["reasoning_effort"] = thinking_level

        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        if response_format:
            payload["response_format"] = response_format

        # 3. Headers
        effective_api_key = (
            api_key
            or self.api_key
            or os.getenv("CUSTOM_PROVIDER_API_KEY", "")
        )

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            **self.default_headers,
        }
        if effective_api_key:
            headers["Authorization"] = f"Bearer {effective_api_key}"

        if extra_headers:
            headers.update(extra_headers)

        # 4. Eksekusi HTTP request via centralized execute_with_cancellation
        async def _call_api() -> httpx.Response:
            async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
                return await client.post(url, headers=headers, json=payload)

        try:
            resp = await self.execute_with_cancellation(_call_api, token=token, caller_name="CustomAdapter")
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"[CustomAdapter] HTTP {e.response.status_code} dari {url}: {e.response.text}"
            )
            raise e
        except httpx.RequestError as e:
            logger.error(f"[CustomAdapter] Network error saat mengakses {url}: {e}")
            raise e

        # 5. Ekstraksi konten, reasoning, dan metadata
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        raw_content = message.get("content") or ""
        reasoning = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or ""
        )

        # Jika reasoning masih kosong tapi terdapat tag <think> di content
        if not reasoning and "<think>" in raw_content and "</think>" in raw_content:
            think_match = re.search(r"<think>(.*?)</think>", raw_content, flags=re.DOTALL)
            if think_match:
                reasoning = think_match.group(1).strip()
                raw_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

        tool_calls = message.get("tool_calls") or []

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Ekstraksi token cache hit dari berbagai format penyedia
        cached_prompt_tokens = (
            usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            or usage.get("prompt_cache_hit_tokens", 0)
            or usage.get("cache_creation_input_tokens", 0)
            or 0
        )

        return {
            "content": raw_content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
            "prompt_tokens": prompt_tokens,
            "cached_prompt_tokens": cached_prompt_tokens,
            "completion_tokens": completion_tokens,
            "model": data.get("model", model),
            "provider": provider,
        }


# Alias kemudahan
CustomProviderAdapter = CustomAdapter

__all__ = [
    "CustomAdapter",
    "CustomProviderAdapter",
]
