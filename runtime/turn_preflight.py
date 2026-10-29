"""
Fase 1 Turn Engine (Preflight) untuk LegionADK Runtime.

Menyediakan gerbang pemeriksaan pra-giliran yang deterministik, sinkron,
dan murni (tanpa efek samping I/O) sebelum token dibelanjakan.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
)

# pyrefly: ignore[missing-import]
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from LegionADK.core.llm_spec import CancellationToken
    from LegionADK.governance.smart_guardian import SmartGuardian
    from LegionADK.runtime.turn_state import SessionState, TurnPolicy


# ============================================================================
# 1. PreflightResult Contract
# ============================================================================

class PreflightResult(BaseModel):
    """
    Hasil evaluasi gerbang Fase 1 (Preflight) sebelum giliran agen dieksekusi.
    Frozen dan immutable.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    can_proceed: bool = Field(
        ...,
        description="Menandakan apakah giliran diizinkan untuk dieksekusi (True jika status='READY')",
    )
    status: Literal[
        "READY",
        "CANCELLED",
        "TERMINATED",
        "BLOCKED_LOOP",
        "EXCEEDED_MAX_TURNS",
        "BUDGET_EXCEEDED",
    ] = Field(
        ...,
        description="Status hasil pra-pemeriksaan gerbang preflight",
    )
    reason: str = Field(
        default="",
        description="Alasan penolakan atau penjelasan status bila tidak siap",
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> "PreflightResult":
        """Menegakkan konsistensi: can_proceed wajib sama dengan (status == 'READY')."""
        if self.can_proceed != (self.status == "READY"):
            raise ValueError(
                f"Inkonsistensi PreflightResult: can_proceed={self.can_proceed} "
                f"tidak sesuai dengan status='{self.status}' (harus True jika dan hanya jika status='READY')."
            )
        return self


# ============================================================================
# 2. Synchronous Pure Preflight Gate
# ============================================================================

def run_preflight(
    state: SessionState,
    policy: TurnPolicy,
    token: Optional[CancellationToken] = None,
    guardian: Optional[SmartGuardian] = None,
) -> PreflightResult:
    """
    Menjalankan pemeriksaan pra-giliran (Fase 1 Preflight) secara sinkron dan murni.

    Urutan evaluasi tetap:
    1. Token batal (CANCELLED)
    2. Status terminasi SmartGuardian (TERMINATED)
    3. Ambang batas putaran kosong berturut-turut (BLOCKED_LOOP)
    4. Ambang batas jumlah giliran maksimum (EXCEEDED_MAX_TURNS)
    5. Kuota token sesi (BUDGET_EXCEEDED)

    Fungsi ini bersifat murni: tidak memutasi objek apa pun dan tidak memanggil
    metode pencatat state internal.
    """
    # 1. Token batal: gagal pertama menang
    if token is not None and token.is_cancelled:
        raw_reason = token.reason
        reason_str = str(raw_reason).strip() if raw_reason else ""
        reason = (
            reason_str
            if reason_str
            else "Giliran dibatalkan sebelum dimulai: token pembatalan aktif."
        )
        return PreflightResult(
            can_proceed=False,
            status="CANCELLED",
            reason=reason,
        )

    # 2. Guardian terminate: get_session_level dipanggil tepat satu kali
    if guardian is not None and guardian.get_session_action(state.session_id) == "TERMINATE":
        return PreflightResult(
            can_proceed=False,
            status="TERMINATED",
            reason=(
                f"Sesi '{state.session_id}' telah diterminasi oleh SmartGuardian: "
                f"level eskalasi mencapai TERMINATE."
            ),
        )
    # 3. Breaker putaran
    if state.consecutive_empty_turns >= policy.max_consecutive_empty_turns:
        return PreflightResult(
            can_proceed=False,
            status="BLOCKED_LOOP",
            reason=(
                f"Giliran ditolak: {state.consecutive_empty_turns} giliran kosong berturut-turut "
                f"telah mencapai ambang batas {policy.max_consecutive_empty_turns}."
            ),
        )

    # 4. Batas giliran
    if state.turn_count >= policy.max_turns:
        return PreflightResult(
            can_proceed=False,
            status="EXCEEDED_MAX_TURNS",
            reason=(
                f"Giliran ditolak: jumlah giliran ({state.turn_count}) "
                f"telah mencapai batas maksimum ({policy.max_turns})."
            ),
        )

    # 5. Kuota token sesi (hanya total_prompt_tokens + total_completion_tokens)
    if policy.session_token_budget is not None:
        used_tokens = state.total_prompt_tokens + state.total_completion_tokens
        if used_tokens >= policy.session_token_budget:
            return PreflightResult(
                can_proceed=False,
                status="BUDGET_EXCEEDED",
                reason=(
                    f"Giliran ditolak: total token yang digunakan ({used_tokens}) "
                    f"telah mencapai kuota sesi ({policy.session_token_budget})."
                ),
            )

    # 6. Semua pemeriksaan lolos
    return PreflightResult(
        can_proceed=True,
        status="READY",
        reason="",
    )


__all__ = ["PreflightResult", "run_preflight"]
