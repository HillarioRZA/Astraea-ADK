"""
Anthropic REST Adapter untuk LegionADK.
Mendukung model Claude (Opus, Sonnet, Haiku) dengan Prompt Caching dan Extended Thinking.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

# pyrefly: ignore[missing-import]
import httpx

logger = logging.getLogger("LegionADK.gateway.adapter.anthropic")

from LegionADK.gateway.adapter.base_adapter import BaseLLMAdapter

import copy

if TYPE_CHECKING:
    from LegionADK.core.llm_spec import CancellationToken


def _normalize_content_to_blocks(content: Any) -> List[Dict[str, Any]]:
    """Menormalisasi konten pesan ke format list of content blocks Anthropic."""
    if isinstance(content, list):
        return list(content)
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    if content is None:
        return []
    return [{"type": "text", "text": str(content)}]


def _merge_two_messages(prev: Dict[str, Any], curr: Dict[str, Any]) -> Dict[str, Any]:
    """Menggabungkan dua pesan berurutan yang memiliki role sama menjadi satu pesan tunggal."""
    prev_c = prev.get("content", "")
    curr_c = curr.get("content", "")

    if isinstance(prev_c, str) and isinstance(curr_c, str):
        if prev_c and curr_c:
            merged_content: Any = f"{prev_c}\n\n{curr_c}"
        else:
            merged_content = prev_c or curr_c
    else:
        merged_content = _normalize_content_to_blocks(prev_c) + _normalize_content_to_blocks(curr_c)

    res = dict(prev)
    res["content"] = merged_content
    return res


def _merge_consecutive_same_role_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Menggabungkan pesan-pesan berurutan yang memiliki role identik
    (misal user + user atau assistant + assistant) menjadi satu pesan tunggal.
    Menegakkan invarian alternasi role ketat pada Anthropic Messages API.
    """
    if not messages:
        return []

    merged: List[Dict[str, Any]] = []
    for msg in messages:
        if not merged:
            merged.append(dict(msg))
            continue

        if merged[-1].get("role") == msg.get("role"):
            merged[-1] = _merge_two_messages(merged[-1], msg)
        else:
            merged.append(dict(msg))

    return merged


class AnthropicAdapter(BaseLLMAdapter):
    """Adapter untuk model Anthropic Claude via REST API murni."""

    DEFAULT_BASE_URL = "https://api.anthropic.com/v1"

    async def complete(
        self,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: str = "",
        model: str = "",
        api_key: str = "",
        base_url: str = "",
        temperature: float = 0.2,
        max_tokens: Optional[int] = 4096,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        images: Optional[List[str]] = None,
        thinking_level: str = "medium",
        token: Optional[CancellationToken] = None,
    ) -> Dict[str, Any]:
        """Menjalankan pemanggilan completion teks via REST endpoint /messages."""
        target_base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        url = f"{target_base_url}/messages"

        # 1. Bangun messages dan pisahkan system role
        anthropic_messages: List[Dict[str, Any]] = []
        extracted_system: List[str] = []

        if system_prompt:
            extracted_system.append(system_prompt)

        def _format_images(imgs: List[str]) -> List[Dict[str, Any]]:
            blocks = []
            for img in imgs:
                if img.startswith("data:"):
                    header, data_b64 = img.split(",", 1)
                    media_type = header.split(";")[0].replace("data:", "")
                else:
                    media_type = "image/jpeg"
                    data_b64 = img
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data_b64,
                    }
                })
            return blocks

        if messages:
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    extracted_system.append(str(content))
                elif role == "tool":
                    # Map OpenAI format role 'tool' ke Anthropic tool_result block di role 'user'
                    tool_call_id = msg.get("tool_call_id", "")
                    anthropic_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "content": str(content),
                            }
                        ],
                    })
                elif role == "assistant" and msg.get("tool_calls"):
                    blocks: List[Dict[str, Any]] = []
                    if content:
                        blocks.append({"type": "text", "text": str(content)})
                    for tc in msg.get("tool_calls", []):
                        fn = tc.get("function", {})
                        fn_args = fn.get("arguments", "{}")
                        parsed_args = json.loads(fn_args) if isinstance(fn_args, str) else fn_args
                        blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": parsed_args if isinstance(parsed_args, dict) else {},
                        })
                    anthropic_messages.append({"role": "assistant", "content": blocks})
                else:
                    anthropic_messages.append({"role": role, "content": content})

            # Lampirkan images ke pesan user terakhir jika ada
            if images and anthropic_messages:
                img_blocks = _format_images(images)
                for idx in range(len(anthropic_messages) - 1, -1, -1):
                    if anthropic_messages[idx]["role"] == "user":
                        orig = anthropic_messages[idx]["content"]
                        if isinstance(orig, str):
                            anthropic_messages[idx]["content"] = img_blocks + [{"type": "text", "text": orig}]
                        elif isinstance(orig, list):
                            anthropic_messages[idx]["content"] = img_blocks + orig
                        break
        elif prompt:
            user_content: Union[str, List[Dict[str, Any]]] = prompt
            if images:
                content_parts: List[Dict[str, Any]] = _format_images(images)
                content_parts.append({"type": "text", "text": prompt})
                user_content = content_parts

            anthropic_messages.append({"role": "user", "content": user_content})

        # Gabungkan pesan berurutan yang memiliki role identik sebelum dikirim ke Anthropic
        anthropic_messages = _merge_consecutive_same_role_messages(anthropic_messages)

        # Invarian Anthropic: pesan pertama wajib ber-role 'user'
        if anthropic_messages and anthropic_messages[0]["role"] != "user":
            anthropic_messages.insert(0, {"role": "user", "content": "(Melanjutkan sesi percakapan)"})

        # 2. Siapkan payload request
        effective_max_tokens = max_tokens if (max_tokens is not None and max_tokens > 0) else 4096
        payload: Dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
        }

        # Prompt cache pada system prompt dengan pemisahan LEGION_CACHE_BOUNDARY
        if extracted_system:
            combined_system = "\n\n".join(extracted_system)
            if "<!-- LEGION_CACHE_BOUNDARY -->" in combined_system:
                parts = combined_system.split("<!-- LEGION_CACHE_BOUNDARY -->", 1)
                static_part = parts[0].strip()
                dynamic_part = parts[1].strip()
                system_blocks: List[Dict[str, Any]] = [
                    {
                        "type": "text",
                        "text": static_part,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                if dynamic_part:
                    system_blocks.append({
                        "type": "text",
                        "text": dynamic_part,
                    })
                payload["system"] = system_blocks
            else:
                payload["system"] = [
                    {
                        "type": "text",
                        "text": combined_system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]

        # Konfigurasi thinking_level untuk model Anthropic
        if thinking_level and thinking_level in ("low", "medium", "high"):
            budget = 1024 if thinking_level == "low" else (4096 if thinking_level == "medium" else 8192)
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            payload["temperature"] = 1.0
            if effective_max_tokens <= budget:
                payload["max_tokens"] = budget + 2048

        # Konversi tools jika dalam format OpenAI
        if tools:
            anthropic_tools = []
            for t in tools:
                if "function" in t:
                    fn = t["function"]
                    anthropic_tools.append({
                        "name": fn.get("name"),
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                    })
                else:
                    anthropic_tools.append(t)
            payload["tools"] = anthropic_tools

            if tool_choice:
                if isinstance(tool_choice, str):
                    if tool_choice in ("auto", "any", "none"):
                        payload["tool_choice"] = {"type": tool_choice}
                elif isinstance(tool_choice, dict):
                    payload["tool_choice"] = tool_choice

        # 3. Headers
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
        }
        if api_key:
            headers["x-api-key"] = api_key

        # 4. Eksekusi HTTP request dengan CancellationToken terpusat & trust_env=False
        async def _call_api() -> httpx.Response:
            async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
                return await client.post(url, headers=headers, json=payload)

        try:
            resp: httpx.Response = await self.execute_with_cancellation(
                _call_api, token=token, caller_name="AnthropicAdapter"
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"[AnthropicAdapter] HTTP {e.response.status_code} dari {url}: {e.response.text}"
            )
            raise e
        except httpx.RequestError as e:
            logger.error(f"[AnthropicAdapter] Network error saat mengakses {url}: {e}")
            raise e

        # 5. Ekstraksi konten, reasoning, dan tool_calls
        content_blocks = data.get("content", [])
        text_parts: List[str] = []
        thinking_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        for block in content_blocks:
            b_type = block.get("type")
            if b_type == "text":
                text_parts.append(block.get("text", ""))
            elif b_type == "thinking":
                thinking_parts.append(block.get("thinking", ""))
            elif b_type == "tool_use":
                tool_calls.append({
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })

        content = "\n".join(text_parts)
        reasoning = "\n".join(thinking_parts)

        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        cached_prompt_tokens = usage.get("cache_read_input_tokens", 0)

        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
            "prompt_tokens": prompt_tokens,
            "cached_prompt_tokens": cached_prompt_tokens,
            "completion_tokens": completion_tokens,
            "model": data.get("model", model),
            "provider": "anthropic",
        }
