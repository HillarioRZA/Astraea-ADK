"""
Base Interface untuk LLM Adapters.
Menetapkan kontrak pemanggilan REST/SSE tanpa framework pihak ketiga.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from LegionADK.core.llm_spec import CancellationToken


class BaseLLMAdapter(ABC):
    """Kontrak abstrak dasar untuk seluruh adapter LLM murni native."""

    @abstractmethod
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
    ) -> Dict[str, Any]:
        """Menjalankan pemanggilan completion teks secara asinkron.

        Argumen:
            prompt: Teks prompt tunggal (alternatif jika messages tidak diisi).
            messages: Riwayat percakapan chat lengkap [{"role": "user", "content": ...}].
            system_prompt: Instruksi sistem / compiled identity agen.
            model: ID model LLM (e.g. 'deepseek-v4.1-flash', 'claude-opus-5-5').
            api_key: Kunci otentikasi API penyedia.
            base_url: Endpoint URL kustom (opsional).
            temperature: Suhu sampling (kreativitas respons).
            max_tokens: Batas maksimum token keluaran.
            tools: Daftar spesifikasi tools / function definitions yang diizinkan.
            tool_choice: Kontrol pemanggilan tool ('auto', 'required', 'none', atau objek spesifik).
            response_format: Konfigurasi JSON schema / structured output (e.g. {"type": "json_object"}).
            images: Daftar URL berkas atau data gambar base64 untuk model vision.
            thinking_level: Tingkat kedalaman penalaran untuk model reasoning (low, medium, high).
            token: CancellationToken untuk menghentikan/membatalkan request HTTP di tengah jalan.

        Mengembalikan kamus:
            {
                "content": str,
                "reasoning": str,
                "tool_calls": List[Dict[str, Any]],
                "prompt_tokens": int,
                "cached_prompt_tokens": int,
                "completion_tokens": int,
                "model": str,
                "provider": str,
            }
        """
        pass

    async def execute_with_cancellation(
        self,
        coro_func: Any,
        token: Optional[CancellationToken] = None,
        caller_name: str = "Adapter",
    ) -> Any:
        """
        Helper kanonis untuk mengeksekusi coroutine panggilan API dengan pemantauan CancellationToken.
        Menjamin thread-safe cancellation dan penanganan ON_HOLD (wait_if_paused) secara bersih.
        """
        return await execute_with_cancellation(coro_func, token=token, caller_name=caller_name)


import asyncio

async def execute_with_cancellation(
    coro_func: Any,
    token: Optional[CancellationToken] = None,
    caller_name: str = "Adapter",
) -> Any:
    """
    Helper kanonis untuk mengeksekusi coroutine panggilan API dengan pemantauan CancellationToken.
    Menjamin thread-safe cancellation dan penanganan ON_HOLD (wait_if_paused) secara bersih.
    """
    if token is not None:
        await token.wait_if_paused()
        token.raise_if_cancelled()

        call_task = asyncio.create_task(coro_func())
        cancel_task = asyncio.create_task(token.wait_until_cancelled())
        done, pending = await asyncio.wait(
            [call_task, cancel_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for p in pending:
            p.cancel()

        if cancel_task in done:
            call_task.cancel()
            raise asyncio.CancelledError(
                token.reason or f"{caller_name}: Call cancelled by CancellationToken"
            )
        return call_task.result()
    else:
        return await coro_func()


__all__ = [
    "BaseLLMAdapter",
    "execute_with_cancellation",
]
