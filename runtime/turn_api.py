"""
Fase 3-4 Turn Engine (Panggilan Model) untuk LegionADK Runtime.

Menangani eksekusi pemanggilan model LLM via ModelRouter yang dibungkus
dengan watchdog timeout serta mengekstrak respons mentah adapter
menjadi kontrak ApiOutcome yang frozen.
"""

from __future__ import annotations

import asyncio
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Tuple,
)

# pyrefly: ignore[missing-import]
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from LegionADK.core.llm_spec import CancellationToken
    from LegionADK.gateway.model_router import ModelRouter
    from LegionADK.runtime.turn_request import TurnRequest


# ============================================================================
# 1. Custom Turn Timeout Error
# ============================================================================

class TurnTimeoutError(TimeoutError):
    """
    Eksepsi khusus untuk menandai pemanggilan model yang melampaui batas waktu timeout giliran.
    Memungkinkan pemanggil membedakan timeout giliran dari kegagalan router biasa.
    """


# ============================================================================
# 2. ApiOutcome Contract
# ============================================================================

class ApiOutcome(BaseModel):
    """
    Kontrak beku hasil keluaran pemanggilan model LLM dari layer gateway.
    Frozen dan immutable.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str = Field(
        default="",
        description="Konten teks respons utama yang dihasilkan model",
    )
    reasoning: str = Field(
        default="",
        description="Teks penalaran internal (thinking/reasoning process) dari model",
    )
    tool_calls: Tuple[Dict[str, Any], ...] = Field(
        default=(),
        description="Daftar pemanggilan tool yang diajukan oleh model (wadah immutable tuple)",
    )
    prompt_tokens: int = Field(
        default=0,
        description="Jumlah token prompt yang dikonsumsi dalam panggilan ini",
    )
    cached_tokens: int = Field(
        default=0,
        description="Jumlah token prompt ter-cache yang dihemat dalam panggilan ini",
    )
    completion_tokens: int = Field(
        default=0,
        description="Jumlah token completion yang dihasilkan dalam panggilan ini",
    )
    model: str = Field(
        default="",
        description="Nama model LLM aktual yang merespons permintaan",
    )
    provider: str = Field(
        default="",
        description="Nama provider aktual yang melayani permintaan",
    )


# ============================================================================
# 3. Model Invocation Gate (Watchdog Timeout)
# ============================================================================

async def invoke_llm(
    router: ModelRouter,
    request: TurnRequest,
    *,
    token: Optional[CancellationToken] = None,
    timeout_seconds: float = 60.0,
) -> ApiOutcome:
    """
    Mengeksekusi pemanggilan model ter-rute dengan pengawasan watchdog timeout.

    Mengubah asyncio.TimeoutError menjadi TurnTimeoutError dengan deskripsi batas detik.
    Meneruskan asyncio.CancelledError dan error router lainnya apa adanya.
    Mengekstrak dict mentah keluaran adapter menjadi kontrak ApiOutcome beku.
    """
    tools_arg = list(request.tools) if request.tools else None

    try:
        raw = await asyncio.wait_for(
            router.execute_routed_completion(
                messages=list(request.messages),
                system_prompt=request.system_prompt,
                primary_provider=request.route.provider,
                primary_model=request.route.model,
                fallback_models=list(request.route.fallback_models),
                temperature=request.route.temperature,
                max_tokens=request.route.max_tokens,
                tools=tools_arg,
                thinking_level=request.route.thinking_level,
                token=token,
            ),
            timeout=timeout_seconds,
        )
    except (asyncio.TimeoutError, TimeoutError) as exc:
        raise TurnTimeoutError(
            f"Pemanggilan model melampaui batas waktu timeout ({timeout_seconds} detik)."
        ) from exc

    content = raw.get("content") or ""
    reasoning = raw.get("reasoning") or ""
    tool_calls = tuple(raw.get("tool_calls") or ())
    prompt_tokens = raw.get("prompt_tokens") or 0
    cached_tokens = raw.get("cached_prompt_tokens") or 0
    completion_tokens = raw.get("completion_tokens") or 0
    model = raw.get("model") or ""
    provider = raw.get("provider") or ""

    return ApiOutcome(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
        model=model,
        provider=provider,
    )


__all__ = ["ApiOutcome", "TurnTimeoutError", "invoke_llm"]
