"""
Core LLM Specifications, Message Envelopes, and Concurrency Controls for LegionADK.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union
from types import MappingProxyType
# pyrefly: ignore[missing-import]
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


def _deep_freeze(obj: Any) -> Any:
    """Membekukan dictionary dan list secara rekursif menjadi MappingProxyType dan tuple."""
    if isinstance(obj, (dict, MappingProxyType)):
        return MappingProxyType({k: _deep_freeze(v) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return tuple(_deep_freeze(x) for x in obj)
    return obj


def _deep_unfreeze(obj: Any) -> Any:
    """Mengonversi MappingProxyType dan tuple kembali ke dict dan list murni untuk serialisasi."""
    if isinstance(obj, (MappingProxyType, dict)):
        return {k: _deep_unfreeze(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return [_deep_unfreeze(x) for x in obj]
    return obj


# ============================================================================
# 1. LLMSpec Contract & compile_identity()
# ============================================================================

class LLMSpec(BaseModel):
    """
    Kontrak spesifikasi agen LLM:
    Menyimpan @handle, nama peran, teks Layer 1 Soul, Layer 2 SOP,
    daftar izin tools (immutable tuple), dan konfigurasi model (read-only mapping).
    """
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    handle: str = Field(
        ...,
        description="Identitas @handle agen, contoh: '@supervisor' atau '@cto'"
    )
    agent_name: str = Field(
        ...,
        description="Nama peran agen dalam sistem"
    )
    soul_prompt: str = Field(
        ...,
        description="Layer 1: Identity & Soul (karakter, prinsip inti, tone bicara)"
    )
    system_prompt: str = Field(
        ...,
        description="Layer 2: Core SOP & Playbook (prosedur operasional standar)"
    )
    tool_permissions: Optional[Tuple[str, ...]] = Field(
        default=None,
        description="Daftar izin tools yang diakses oleh agen (immutable tuple)"
    )
    model_settings: Mapping[str, Any] = Field(
        default_factory=dict,
        description="Konfigurasi parameter model (read-only mapping)"
    )

    @field_validator("handle")
    @classmethod
    def validate_handle(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned.startswith("@"):
            raise ValueError(f"Handle agen harus diawali tanda '@', contoh: '@{cleaned}'")
        return cleaned

    @field_validator("tool_permissions", mode="before")
    @classmethod
    def validate_tool_permissions(cls, v: Any) -> Optional[Tuple[str, ...]]:
        if v is None:
            return None
        if isinstance(v, (list, tuple, set)):
            return tuple(str(x) for x in v)
        return v

    @field_validator("model_settings", mode="after")
    @classmethod
    def freeze_model_settings(cls, v: Any) -> Mapping[str, Any]:
        return _deep_freeze(v)

    @field_serializer("model_settings", when_used="always")
    def serialize_model_settings(self, v: Any) -> Dict[str, Any]:
        return _deep_unfreeze(v)

    def compile_identity(self) -> str:
        """
        Menggabungkan Layer 1 (Identity/Soul), Layer 2 (Core SOP/Playbook),
        serta Layer 3 (Tool Permissions) yang diurutkan berdasarkan abjad
        menjadi satu blok teks statis yang kanonis dan 100% byte-stable.
        """
        sorted_tools = sorted(self.tool_permissions) if self.tool_permissions else []
        tools_content = "\n".join(f"- `{tool}`" for tool in sorted_tools) if sorted_tools else "- NONE"

        parts = [
            f"=== ROLE IDENTIFIER: {self.agent_name} ({self.handle}) ===".strip(),
            "",
            "=== LAYER 1: IMMUTABLE SOUL & PERSONA ===",
            self.soul_prompt.strip(),
            "",
            "=== LAYER 2: STATIC OPERATIONAL SOP & RULES ===",
            self.system_prompt.strip(),
            "",
            "=== AUTHORIZED TOOLS (DETERMINISTIC IN-MEMORY ORDER) ===",
            tools_content,
        ]
        return "\n".join(parts).strip()


def compile_identity(spec: LLMSpec) -> str:
    """
    Helper function top-level untuk mengompilasi identitas statis dari sebuah LLMSpec.
    """
    return spec.compile_identity()


# ============================================================================
# 2. MoA Metadata & MessageEnvelope Contract
# ============================================================================

class MoAMetadata(BaseModel):
    """
    Kontrak metadata terstruktur untuk Mixture-of-Agents (MoA).
    Melacak posisi agen dalam diskusi (proposer/aggregator), layer,
    putaran diskusi, latensi, statistik token, dan draf usulan.
    """
    model_config = ConfigDict(extra="allow", frozen=True, arbitrary_types_allowed=True)

    moa_role: Literal["standalone", "proposer", "aggregator"] = Field(
        default="standalone",
        description="Peran agen dalam sesi saat ini: standalone, proposer, atau aggregator"
    )
    layer: int = Field(
        default=1,
        ge=1,
        description="Indeks layer MoA (Layer 1..N untuk proposer, Layer akhir untuk aggregator)"
    )
    round_index: int = Field(
        default=1,
        ge=1,
        description="Putaran diskusi antar-agen sebelum masuk ke tahap sintesis aggregator"
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Waktu eksekusi model dalam milidetik"
    )
    token_stats: Mapping[str, int] = Field(
        default_factory=dict,
        description="Statistik konsumsi token (read-only mapping)"
    )
    proposals: Tuple[Any, ...] = Field(
        default_factory=tuple,
        description="Kumpulan usulan dari agen proposer (immutable deep-frozen tuple)"
    )

    @field_validator("proposals", mode="before")
    @classmethod
    def validate_proposals(cls, v: Any) -> Tuple[Any, ...]:
        if v is None:
            return ()
        if isinstance(v, (list, tuple)):
            return tuple(_deep_freeze(x) for x in v)
        return v

    @field_validator("token_stats", mode="after")
    @classmethod
    def freeze_token_stats(cls, v: Any) -> Mapping[str, int]:
        return _deep_freeze(v)

    @field_serializer("proposals", when_used="always")
    def serialize_proposals(self, v: Any) -> List[Any]:
        return _deep_unfreeze(v)

    @field_serializer("token_stats", when_used="always")
    def serialize_token_stats(self, v: Any) -> Dict[str, Any]:
        return _deep_unfreeze(v)


class MessageEnvelope(BaseModel):
    """
    Membungkus setiap pesan yang dikirim oleh manusia maupun agen dengan
    ID amplop unik (128-bit UUID4), pengirim, target, isi pesan, lampiran, dan metadata.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    envelope_id: str = Field(
        default_factory=lambda: f"msg_{uuid.uuid4().hex}",
        description="ID unik amplop pesan 128-bit UUID4 untuk pelacakan deterministik bebas collision"
    )
    sender: str = Field(
        ...,
        description="Pengirim pesan (identitas agen @handle atau user)"
    )
    target: List[str] = Field(
        ...,
        description="Target penerima: satu atau lebih @handle spesifik (e.g. ['@cto', '@pm']) atau ['@all']"
    )
    content: str = Field(
        ...,
        description="Isi teks pesan"
    )
    attachments: List[str] = Field(
        default_factory=list,
        description="Daftar path berkas atau dokumen lampiran"
    )
    reply_to: Optional[str] = Field(
        default=None,
        description="ID envelope_id pesan yang dibalas (jika ada)"
    )
    metadata: MoAMetadata = Field(
        default_factory=MoAMetadata,
        description="Metadata MoA terstruktur (moa_role, layer, round_index, latency_ms, token_stats)"
    )

    @field_validator("target", mode="before")
    @classmethod
    def validate_target(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            items = [item.strip() for item in v.split(",") if item.strip()]
        elif isinstance(v, (list, tuple, set)):
            items = [str(item).strip() for item in v if str(item).strip()]
        else:
            raise ValueError(
                f"target harus berupa string '@handle' atau list of '@handle', diterima: {type(v).__name__}"
            )

        if not items:
            raise ValueError("target tidak boleh kosong, minimal satu '@handle' atau '@all'")

        validated_targets: List[str] = []
        for item in items:
            if not item.startswith("@"):
                raise ValueError(
                    f"Setiap target harus diawali dengan tanda '@', contoh: '@handle' atau '@all', diterima: '{item}'"
                )
            validated_targets.append(item)

        # Hapus duplikat target dengan tetap mempertahankan urutan asli
        return list(dict.fromkeys(validated_targets))

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, v: Any) -> MoAMetadata:
        if v is None:
            return MoAMetadata()
        if isinstance(v, MoAMetadata):
            return v
        if isinstance(v, dict):
            return MoAMetadata(**v)
        raise ValueError(f"metadata harus berupa dict atau MoAMetadata, diterima: {type(v).__name__}")


def contract_message(
    sender: str,
    target: Union[str, List[str]],
    content: str,
    attachments: Optional[List[str]] = None,
    reply_to: Optional[str] = None,
    metadata: Optional[Union[Dict[str, Any], MoAMetadata]] = None,
    envelope_id: Optional[str] = None,
) -> MessageEnvelope:
    """
    Membungkus setiap pesan yang dikirim oleh manusia maupun agen dengan metadata
    pengirim, target, lampiran berkas, dan metadata MoA menggunakan validasi ketat Pydantic v2.
    """
    kwargs: Dict[str, Any] = {
        "sender": sender,
        "target": target,
        "content": content,
        "attachments": attachments or [],
        "reply_to": reply_to,
        "metadata": metadata or MoAMetadata(),
    }
    if envelope_id is not None:
        kwargs["envelope_id"] = envelope_id
    return MessageEnvelope(**kwargs)


# ============================================================================
# 3. CancellationToken & check_if_paused()
# ============================================================================

class CancellationToken:
    """
    Token pembatalan dan penangguhan berbasis asyncio.Event di memori
    untuk menghentikan atau menunda (ON_HOLD) putaran agen secara instan dari luar.
    """

    def __init__(self, reason: Optional[str] = None) -> None:
        self._cancel_event = asyncio.Event()
        self._resume_event = asyncio.Event()
        self._resume_event.set()  # Default: tidak di-pause (berjalan normal)
        self._paused: bool = False
        self._reason: Optional[str] = reason

        if reason is not None:
            self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        """Mengembalikan True jika token telah dibatalkan secara permanen."""
        return self._cancel_event.is_set()

    @property
    def is_paused(self) -> bool:
        """Mengembalikan True jika status agen sedang ditangguhkan (ON_HOLD)."""
        return self._paused

    @property
    def reason(self) -> Optional[str]:
        """Mengembalikan alasan penghentian atau penangguhan."""
        return self._reason

    def cancel(self, reason: str = "Execution cancelled") -> None:
        """
        Menghentikan putaran agen secara instan dan membebaskan antrean pause.
        """
        self._reason = reason
        self._cancel_event.set()
        self._resume_event.set()

    def pause(self, reason: str = "Agent paused (ON_HOLD)") -> None:
        """
        Menandai status agen sebagai ON_HOLD/tertunda di memori.
        Hanya menangguhkan eksekusi, tidak membatalkan token secara permanen.
        """
        if self._cancel_event.is_set():
            return  # Token yang sudah dibatalkan tidak bisa di-pause
        self._paused = True
        self._reason = reason
        self._resume_event.clear()

    def resume(self) -> None:
        """
        Melanjutkan kembali eksekusi agen dari status ON_HOLD/tertunda.
        Tidak berpengaruh jika token telah dibatalkan permanen.
        """
        if self._cancel_event.is_set():
            return  # Token yang sudah dibatalkan permanen tidak bisa di-resume
        self._paused = False
        self._reason = None
        self._resume_event.set()

    def check_if_paused(self) -> bool:
        """
        Memeriksa apakah eksekusi agen sedang dalam kondisi ditangguhkan (ON_HOLD).
        """
        return self._paused

    async def wait_if_paused(self) -> None:
        """
        Menangguhkan coroutine secara non-blocking saat status ON_HOLD aktif
        hingga resume() dipanggil atau token dibatalkan.
        """
        while self._paused and not self._cancel_event.is_set():
            await self._resume_event.wait()
        self.raise_if_cancelled()

    def raise_if_cancelled(self) -> None:
        """
        Memicu asyncio.CancelledError HANYA jika token telah dibatalkan (cancel).
        TIDAK melempar saat status hanya dalam kondisi pause (ON_HOLD).
        """
        if self._cancel_event.is_set():
            raise asyncio.CancelledError(self._reason or "Execution cancelled by CancellationToken")

    async def wait_until_cancelled(self) -> None:
        """
        Helper publik untuk menunggu event pembatalan (dipakai oleh adapter HTTP).
        """
        await self._cancel_event.wait()


def check_if_paused(token: Optional[CancellationToken] = None) -> bool:
    """
    Helper fungsi untuk memeriksa apakah eksekusi agen sedang ditangguhkan / dibatalkan.
    """
    if token is None:
        return False
    return token.check_if_paused()


__all__ = [
    "LLMSpec",
    "compile_identity",
    "MoAMetadata",
    "MessageEnvelope",
    "contract_message",
    "CancellationToken",
    "check_if_paused",
]
