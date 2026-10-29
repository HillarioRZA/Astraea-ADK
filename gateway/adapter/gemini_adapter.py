"""
Google Gemini REST Adapter untuk LegionADK.
Mendukung model Gemini 1.5, Gemini 2.0 Flash, dan Thinking via REST API murni.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

# pyrefly: ignore[missing-import]
import httpx

logger = logging.getLogger("LegionADK.gateway.adapter.gemini")

from LegionADK.gateway.adapter.base_adapter import BaseLLMAdapter

if TYPE_CHECKING:
    from LegionADK.core.llm_spec import CancellationToken


def _merge_consecutive_same_role_contents(
    contents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Menggabungkan turn-turn berurutan yang memiliki role identik pada Gemini
    (misal user + user atau model + model) dengan menyatukan daftar parts-nya.
    Menegakkan aturan alternasi turn ketat pada Google Gemini API.
    """
    if not contents:
        return []

    merged: List[Dict[str, Any]] = []
    for turn in contents:
        if not merged:
            merged.append(dict(turn))
            continue

        if merged[-1].get("role") == turn.get("role"):
            merged[-1] = {
                "role": merged[-1]["role"],
                "parts": list(merged[-1].get("parts", [])) + list(turn.get("parts", [])),
            }
        else:
            merged.append(dict(turn))

    return merged


class GeminiAdapter(BaseLLMAdapter):
    """Adapter untuk Google Gemini via REST API murni tanpa vendor SDK."""

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    async def complete(
        self,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: str = "",
        model: str = "gemini-2.0-flash",
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
    ) -> Dict[str, Any]:
        """Menjalankan pemanggilan generateContent teks via REST API."""
        if token is not None:
            await token.wait_if_paused()
            token.raise_if_cancelled()

        target_base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        # Bersihkan awalan 'models/' jika ada pada nama model
        clean_model = model.replace("models/", "")
        url = f"{target_base_url}/models/{clean_model}:generateContent"

        # 1. Bangun contents sesuai format Gemini (role: 'user' / 'model')
        contents: List[Dict[str, Any]] = []
        system_parts: List[str] = [system_prompt] if system_prompt else []

        if messages:
            for msg in messages:
                role = msg.get("role", "user")
                if role == "assistant":
                    role = "model"
                elif role == "system":
                    content_str = str(msg.get("content", ""))
                    if content_str:
                        system_parts.append(content_str)
                    continue
                elif role == "tool":
                    # Pemetaan pesan hasil tool call OpenAI -> Gemini functionResponse
                    tool_name = msg.get("name") or "unknown_function"
                    raw_resp = msg.get("content", "")
                    if isinstance(raw_resp, str):
                        try:
                            resp_obj = json.loads(raw_resp)
                            if not isinstance(resp_obj, dict):
                                resp_obj = {"result": raw_resp}
                        except Exception:
                            resp_obj = {"result": raw_resp}
                    else:
                        resp_obj = raw_resp if isinstance(raw_resp, dict) else {"result": raw_resp}

                    contents.append({
                        "role": "user",
                        "parts": [{
                            "functionResponse": {
                                "name": tool_name,
                                "response": resp_obj,
                            }
                        }],
                    })
                    continue

                raw_content = msg.get("content", "")
                parts: List[Dict[str, Any]] = []
                if isinstance(raw_content, str):
                    if raw_content:
                        parts.append({"text": raw_content})
                elif isinstance(raw_content, list):
                    for part in raw_content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            parts.append({"text": part.get("text", "")})

                # Jika ada tool_calls dari model/assistant turn sebelumnya
                if role == "model" and msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        args = fn.get("arguments", "{}")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {}
                        parts.append({
                            "functionCall": {
                                "name": fn.get("name"),
                                "args": args,
                            }
                        })

                if parts:
                    contents.append({"role": role, "parts": parts})

            # Lampirkan images ke turn user terakhir jika messages disediakan
            if images and contents:
                for c in reversed(contents):
                    if c.get("role") == "user":
                        for img in images:
                            if img.startswith("data:"):
                                header, data_b64 = img.split(",", 1)
                                mime_type = header.split(";")[0].replace("data:", "")
                            else:
                                mime_type = "image/jpeg"
                                data_b64 = img
                            c["parts"].append({
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": data_b64,
                                }
                            })
                        break

        elif prompt:
            parts: List[Dict[str, Any]] = []
            if images:
                for img in images:
                    if img.startswith("data:"):
                        header, data_b64 = img.split(",", 1)
                        mime_type = header.split(";")[0].replace("data:", "")
                    else:
                        mime_type = "image/jpeg"
                        data_b64 = img
                    parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": data_b64,
                        }
                    })
            parts.append({"text": prompt})
            contents.append({"role": "user", "parts": parts})

        # Gabungkan turn berurutan yang memiliki role sama sebelum dikirim ke Gemini
        contents = _merge_consecutive_same_role_contents(contents)

        # Invarian Gemini: turn pertama dalam contents wajib ber-role 'user'
        if contents and contents[0]["role"] != "user":
            contents.insert(0, {"role": "user", "parts": [{"text": "(Melanjutkan sesi percakapan)"}]})

        # 2. Siapkan payload request
        generation_config: Dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens

        # Thinking config jika thinking_level diaktifkan
        if thinking_level and thinking_level not in ("none", "off"):
            budget = 1024 if thinking_level == "low" else (4096 if thinking_level == "medium" else 8192)
            generation_config["thinkingConfig"] = {"thinkingBudget": budget}

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        # System Instruction (menggabungkan semua system parts)
        combined_system = "\n\n".join(system_parts).strip()
        if combined_system:
            payload["system_instruction"] = {
                "parts": [{"text": combined_system}]
            }

        # Response format (JSON Mode)
        if response_format:
            fmt_type = response_format.get("type")
            if fmt_type in ("json_object", "json"):
                payload["generationConfig"]["responseMimeType"] = "application/json"

        # Konversi tools ke format functionDeclarations Gemini
        if tools:
            declarations: List[Dict[str, Any]] = []
            for t in tools:
                if "function" in t:
                    fn = t["function"]
                    declarations.append({
                        "name": fn.get("name"),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {"type": "OBJECT", "properties": {}}),
                    })
                elif "name" in t:
                    declarations.append(t)
            if declarations:
                payload["tools"] = [{"functionDeclarations": declarations}]

        # 3. Headers
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["x-goog-api-key"] = api_key

        # 4. Eksekusi HTTP request via centralized execute_with_cancellation
        async def _call_api() -> httpx.Response:
            async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
                return await client.post(url, headers=headers, json=payload)

        try:
            resp = await self.execute_with_cancellation(_call_api, token=token, caller_name="GeminiAdapter")
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"[GeminiAdapter] HTTP {e.response.status_code} dari {url}: {e.response.text}"
            )
            raise e
        except httpx.RequestError as e:
            logger.error(f"[GeminiAdapter] Network error saat mengakses {url}: {e}")
            raise e

        # 5. Ekstraksi konten, reasoning, dan function calls
        candidates = data.get("candidates", [])
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        if candidates:
            first_candidate = candidates[0]
            cand_content = first_candidate.get("content", {})
            parts_list = cand_content.get("parts", [])

            for idx, part in enumerate(parts_list):
                if "text" in part:
                    # Tangani part thought jika ada (Gemini 2.0 Flash Thinking)
                    if part.get("thought", False):
                        reasoning_parts.append(part.get("text", ""))
                    else:
                        content_parts.append(part.get("text", ""))

                if "functionCall" in part:
                    fn_call = part["functionCall"]
                    tool_calls.append({
                        "id": f"call_{idx}_{fn_call.get('name')}",
                        "type": "function",
                        "function": {
                            "name": fn_call.get("name"),
                            "arguments": json.dumps(fn_call.get("args", {})),
                        },
                    })

        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)

        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)
        cached_prompt_tokens = usage.get("cachedContentTokenCount", 0)

        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
            "prompt_tokens": prompt_tokens,
            "cached_prompt_tokens": cached_prompt_tokens,
            "completion_tokens": completion_tokens,
            "model": clean_model,
            "provider": "gemini",
        }
