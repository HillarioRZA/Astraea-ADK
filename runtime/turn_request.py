"""
Fase 2 Turn Engine (Perakitan Request) untuk LegionADK Runtime.

Merakit parameter lengkap panggilan model LLM dari kontrak-kontrak yang ada:
- Konfigurasi rute dan inferensi dari spec.model_settings.
- Konteks dinamis deterministik dari metadata envelope.
- System prompt statis + dinamis dan digest dari gateway/cache_boundary.
- Skema deklarasi tool yang diizinkan dari tools/permission_registry.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

# pyrefly: ignore[missing-import]
from pydantic import BaseModel, ConfigDict, Field

from LegionADK.gateway.cache_boundary import assemble_cached_system_prompt_from_spec
from LegionADK.tools.permission_registry import export_permitted_tools

if TYPE_CHECKING:
    from LegionADK.core.llm_spec import LLMSpec, MessageEnvelope


# ============================================================================
# 1. RouteConfig Contract
# ============================================================================

class RouteConfig(BaseModel):
    """
    Konfigurasi perutean model, provider, dan parameter inferensi untuk pemanggilan API.
    Frozen dan immutable.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(
        ...,
        description="Nama penyedia LLM (provider), misal: 'custom', 'anthropic', 'openai'",
    )
    model: str = Field(
        ...,
        description="Nama model LLM yang ditargetkan",
    )
    fallback_models: Tuple[Any, ...] = Field(
        default=(),
        description="Daftar model alternatif bila model utama mengalami kegagalan",
    )
    temperature: float = Field(
        default=0.2,
        description="Suhu sampling untuk inferensi model",
    )
    max_tokens: Optional[int] = Field(
        default=4096,
        description="Batas maksimum token output yang dihasilkan (None jika tanpa batas)",
    )
    thinking_level: str = Field(
        default="medium",
        description="Tingkat kedalaman penalaran internal model (misal: 'low', 'medium', 'high')",
    )


# ============================================================================
# 2. Route Configuration Resolver
# ============================================================================

def resolve_route_config(spec: LLMSpec) -> RouteConfig:
    """
    Mengekstrak konfigurasi rute inferensi dari spec.model_settings secara sinkron dan murni.

    Kunci yang tidak ada menggunakan nilai bawaan:
    - provider: 'custom'
    - model: 'deepseek-v4.1-flash'
    - fallback_models: ()
    - temperature: 0.2
    - max_tokens: 4096
    - thinking_level: 'medium'

    Kunci lain yang ada pada mapping diabaikan tanpa error.
    """
    settings = spec.model_settings or {}

    provider = settings.get("provider", "custom")
    model = settings.get("model", "deepseek-v4.1-flash")
    temperature = settings.get("temperature", 0.2)
    max_tokens = settings.get("max_tokens", 4096)
    thinking_level = settings.get("thinking_level", "medium")

    raw_fallback = settings.get("fallback_models", ())
    if isinstance(raw_fallback, (list, tuple)):
        fallback_models = tuple(raw_fallback)
    elif isinstance(raw_fallback, str):
        fallback_models = (raw_fallback,)
    elif raw_fallback is None:
        fallback_models = ()
    else:
        fallback_models = (raw_fallback,)

    return RouteConfig(
        provider=provider,
        model=model,
        fallback_models=fallback_models,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking_level=thinking_level,
    )


# ============================================================================
# 3. TurnRequest Contract
# ============================================================================

class TurnRequest(BaseModel):
    """
    Kumpulan parameter lengkap dan terstruktur untuk satu panggilan model LLM.
    Frozen dan immutable.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    system_prompt: str = Field(
        ...,
        description="System prompt lengkap hasil penggabungan static prefix dan dynamic context",
    )
    system_digest: str = Field(
        ...,
        description="Digest SHA-256 dari static prefix system prompt untuk verifikasi cache",
    )
    tool_digest: str = Field(
        ...,
        description="Digest SHA-256 dari skema tools yang diizinkan untuk verifikasi cache",
    )
    tools: Tuple[Dict[str, Any], ...] = Field(
        default=(),
        description="Daftar skema tool yang diizinkan untuk model (wadah immutable tuple)",
    )
    messages: Tuple[Dict[str, Any], ...] = Field(
        default=(),
        description="Riwayat percakapan ditambah pesan user dari envelope (wadah immutable tuple)",
    )
    route: RouteConfig = Field(
        ...,
        description="Konfigurasi perutean model, provider, dan parameter inferensi",
    )


# ============================================================================
# 4. Turn Request Assembler
# ============================================================================

def assemble_request(
    spec: LLMSpec,
    envelope: MessageEnvelope,
    history: Sequence[Dict[str, Any]],
    *,
    dynamic_context: str = "",
    context_token_budget: Optional[int] = None,
) -> TurnRequest:
    """
    Merakit seluruh komponen yang diperlukan untuk satu panggilan model (Fase 2).

    Urutan langkah:
    1. Ekstrak RouteConfig dari spec.model_settings.
    2. Susun konteks dinamis secara deterministik tanpa stempel waktu.
    3. Bangun system prompt dan hitung digest via gateway cache_boundary.
    4. Ekspor skema deklarasi tools sesuai format provider (anthropic / openai).
    5. Susun urutan messages dari history ditambah pesan user dari envelope.
    """
    # 1. Rute
    route = resolve_route_config(spec)

    # 2. Konteks dinamis rakitan
    parts: List[str] = [f"Pengirim: {envelope.sender}"]
    if envelope.reply_to is not None:
        parts.append(f"Membalas pesan: {envelope.reply_to}")
    if envelope.attachments:
        parts.append(f"Lampiran: {', '.join(envelope.attachments)}")
    if dynamic_context.strip():
        parts.append(dynamic_context)

    assembled_context = "\n\n".join(parts)

    # 3. System prompt
    system_prompt, system_digest, tool_digest = assemble_cached_system_prompt_from_spec(
        spec,
        dynamic_context=assembled_context,
        max_token_budget=context_token_budget,
    )

    # 4. Tools
    fmt: Literal["anthropic", "openai"] = "anthropic" if route.provider == "anthropic" else "openai"
    tools_list = export_permitted_tools(spec, format=fmt)
    tools = tuple(tools_list)

    # 5. Messages
    # System prompt TIDAK masuk ke messages; ia berjalan terpisah di kolom TurnRequest.system_prompt.
    messages_list = list(history) + [{"role": "user", "content": envelope.content}]
    messages = tuple(messages_list)

    # 6. Kembalikan TurnRequest
    return TurnRequest(
        system_prompt=system_prompt,
        system_digest=system_digest,
        tool_digest=tool_digest,
        tools=tools,
        messages=messages,
        route=route,
    )


__all__ = ["RouteConfig", "TurnRequest", "resolve_route_config", "assemble_request"]
