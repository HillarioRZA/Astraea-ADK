"""
Kontrak murni status sesi dan kebijakan giliran untuk paket runtime/turn LegionADK.

Modul ini mendefinisikan model-model Pydantic v2 frozen yang mengatur:
- SessionState: Keadaan sesi yang dibawa pemanggil antar giliran (in-memory state).
- TurnPolicy: Konfigurasi batas operasional dan timeout eksekusi giliran.
- TurnResult: Hasil terstruktur dari eksekusi satu giliran agen.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

# pyrefly: ignore[missing-import]
from pydantic import BaseModel, ConfigDict, Field

from LegionADK.governance.smart_guardian import GuardianReport


# ============================================================================
# 1. SessionState Contract
# ============================================================================

class SessionState(BaseModel):
    """
    State sesi percakapan yang dibawa pemanggil antar giliran.
    Engine runtime tidak pernah menyimpan atau mempertahankan state ini secara internal.
    Frozen dan immutable.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(
        ...,
        min_length=1,
        description="Identitas unik sesi percakapan (wajib minimal 1 karakter)",
    )
    history: Tuple[Dict[str, Any], ...] = Field(
        default=(),
        description="Riwayat pesan percakapan dalam sesi (wadah immutable tuple)",
    )
    turn_count: int = Field(
        default=0,
        ge=0,
        description="Jumlah giliran (turns) yang telah dieksekusi dalam sesi",
    )
    consecutive_empty_turns: int = Field(
        default=0,
        ge=0,
        description="Jumlah giliran kosong berturut-turut tanpa kemajuan baru",
    )
    total_prompt_tokens: int = Field(
        default=0,
        ge=0,
        description="Akumulasi token prompt yang digunakan sepanjang sesi",
    )
    total_cached_tokens: int = Field(
        default=0,
        ge=0,
        description="Akumulasi token prompt ter-cache yang dihemat sepanjang sesi",
    )
    total_completion_tokens: int = Field(
        default=0,
        ge=0,
        description="Akumulasi token completion yang dihasilkan sepanjang sesi",
    )

    def with_history(self, history: Sequence[Dict[str, Any]]) -> SessionState:
        """
        Membangun instans SessionState baru dengan riwayat pesan yang diperbarui.

        Melakukan salinan wadah dangkal menjadi tuple(history). Dictionary pesan
        di dalamnya tidak disalin dan tidak dibekukan (mengikuti konvensi salin-saat-tulis).
        Kolom lainnya tetap sama dan instans asal tidak berubah.
        """
        return SessionState(
            session_id=self.session_id,
            history=tuple(history),
            turn_count=self.turn_count,
            consecutive_empty_turns=self.consecutive_empty_turns,
            total_prompt_tokens=self.total_prompt_tokens,
            total_cached_tokens=self.total_cached_tokens,
            total_completion_tokens=self.total_completion_tokens,
        )

    def record_turn(
        self,
        prompt_tokens: int,
        cached_tokens: int,
        completion_tokens: int,
        *,
        empty: bool,
    ) -> SessionState:
        """
        Membangun instans SessionState baru dengan statistik giliran terakumulasi.

        Menaikkan turn_count sebesar 1, menambahkan prompt_tokens, cached_tokens,
        dan completion_tokens ke total masing-masing.
        Jika empty bernilai True, consecutive_empty_turns bertambah 1;
        jika empty bernilai False, consecutive_empty_turns direset ke 0.

        Argumen token bernilai negatif akan melempar ValueError sebelum instans dibangun.
        """
        if prompt_tokens < 0 or cached_tokens < 0 or completion_tokens < 0:
            raise ValueError(
                f"Jumlah token tidak boleh negatif: prompt_tokens={prompt_tokens}, "
                f"cached_tokens={cached_tokens}, completion_tokens={completion_tokens}."
            )
        new_empty_turns = (self.consecutive_empty_turns + 1) if empty else 0
        return SessionState(
            session_id=self.session_id,
            history=self.history,
            turn_count=self.turn_count + 1,
            consecutive_empty_turns=new_empty_turns,
            total_prompt_tokens=self.total_prompt_tokens + prompt_tokens,
            total_cached_tokens=self.total_cached_tokens + cached_tokens,
            total_completion_tokens=self.total_completion_tokens + completion_tokens,
        )


# ============================================================================
# 2. TurnPolicy Contract
# ============================================================================

class TurnPolicy(BaseModel):
    """
    Kebijakan dan batas operasional eksekusi giliran pada runtime loop.
    Frozen dan immutable.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_turns: int = Field(
        default=10,
        ge=1,
        description="Batas maksimum giliran yang diizinkan dalam satu eksekusi sesi (wajib >= 1)",
    )
    max_tool_rounds: int = Field(
        default=5,
        ge=1,
        description="Batas maksimum putaran eksekusi tool per giliran (wajib >= 1)",
    )
    api_timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        description="Batas waktu pemanggilan API LLM dalam detik (wajib > 0)",
    )
    context_token_budget: Optional[int] = Field(
        default=None,
        gt=0,
        description="Batas token untuk konteks giliran saat ini (wajib > 0 bila terisi)",
    )
    session_token_budget: Optional[int] = Field(
        default=None,
        gt=0,
        description="Batas total token kumulatif sesi (wajib > 0 bila terisi)",
    )
    max_consecutive_empty_turns: int = Field(
        default=3,
        ge=1,
        description="Batas toleransi giliran kosong berturut-turut sebelum circuit breaker aktif (wajib >= 1)",
    )


# ============================================================================
# 3. TurnResult Contract
# ============================================================================

class TurnResult(BaseModel):
    """
    Hasil akhir terstruktur dari eksekusi satu giliran agen pada runtime.
    Frozen dan immutable.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal[
        "COMPLETED",
        "IN_PROGRESS",
        "BLOCKED_LOOP",
        "EXCEEDED_MAX_TURNS",
        "BUDGET_EXCEEDED",
        "CANCELLED",
        "TERMINATED",
        "ERROR",
    ] = Field(..., description="Status akhir eksekusi giliran")
    content: str = Field(default="", description="Konten teks balasan utama dari agen")
    reasoning: str = Field(default="", description="Penalaran atau thought internal dari model (bila ada)")
    tool_results: Tuple[Dict[str, Any], ...] = Field(
        default=(),
        description="Daftar hasil eksekusi tool pada giliran ini (wadah immutable tuple)",
    )
    prompt_tokens: int = Field(
        default=0,
        ge=0,
        description="Token prompt yang digunakan pada giliran ini",
    )
    cached_tokens: int = Field(
        default=0,
        ge=0,
        description="Token prompt ter-cache pada giliran ini",
    )
    completion_tokens: int = Field(
        default=0,
        ge=0,
        description="Token completion yang dihasilkan pada giliran ini",
    )
    rounds_used: int = Field(
        default=0,
        ge=0,
        description="Jumlah panggilan model yang terjadi dalam giliran ini",
    )
    session_state: SessionState = Field(
        ...,
        description="Keadaan sesi terbaru setelah giliran selesai",
    )
    guardian_report: Optional[GuardianReport] = Field(
        default=None,
        description="Snapshot laporan tata kelola jika terjadi eskalasi atau terminasi sesi",
    )
    message: str = Field(
        default="",
        description="Pesan diagnostik atau catatan tambahan terkait status giliran",
    )


__all__ = ["SessionState", "TurnPolicy", "TurnResult"]
