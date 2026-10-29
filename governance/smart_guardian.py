"""
Smart Guardian Governance Engine for LegionADK.

Agregator sinyal governance dan penentu eskalasi sesi percakapan.

Batas Wewenang Arsitektur:
====================================================================================================
ADK hanya memutuskan status dan memproduksi laporan tata kelola (GuardianReport / GuardianDecision).
Pengiriman laporan ke atasan, eskalasi ke manusia (human-in-the-loop), serta tindakan penghentian
sesi yang sesungguhnya dilakukan oleh layer runtime orkestrator atau SovereignOS sebagai pemanggil.
SmartGuardian:
- BUKAN pemeriksa argumen dan BUKAN eksekutor tool.
- Tidak mengirim pesan ke saluran eksternal mana pun.
- Tidak membatalkan CancellationToken pemanggil.
- Tidak mematikan proses sistem operasi.
- Tidak memanggil atau bergantung pada modul governance lain.
- Bersifat murni deterministik di memori (in-memory state machine).
====================================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
import time
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
)

# pyrefly: ignore[missing-import]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger("LegionADK.governance.smart_guardian")


# ============================================================================
# 1. GovernanceEvent Contract (Pydantic v2 Frozen)
# ============================================================================

class GovernanceEvent(BaseModel):
    """
    Representasi event sinyal tata kelola dari seluruh lapisan runtime dalam satu sesi.
    Deep-frozen dan immutable.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[
        "VERIFICATION_REJECTED",
        "REJECTION_LIMIT_REACHED",
        "CIRCUIT_OPENED",
        "TURN_BREAKER_TRIPPED",
    ] = Field(..., description="Jenis/kategori sinyal governance yang terjadi")
    session_id: str = Field(..., description="Identitas sesi agen percakapan")
    tool_name: Optional[str] = Field(
        default=None,
        description="Nama tool kanonis (dinormalisasi ke lowercase bila diisi)"
    )
    detail: str = Field(default="", description="Informasi kontekstual tambahan atau signature")

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        cleaned = v.strip().lower()
        return cleaned if cleaned else None


# ============================================================================
# 2. GovernancePolicy Contract (Pydantic v2 Frozen)
# ============================================================================

class GovernancePolicy(BaseModel):
    """
    Kebijakan ambang batas toleransi pelanggaran sebelum eskalasi sesi dinaikkan.
    Deep-frozen dan immutable.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_rejections_per_tool: int = Field(
        default=3,
        ge=1,
        description="Batas maksimal penolakan CoV pada satu tool sebelum memicu breach"
    )
    max_circuit_trips_per_session: int = Field(
        default=2,
        ge=1,
        description="Batas maksimal kejadian circuit breaker OPEN per sesi sebelum memicu breach"
    )
    max_turn_breaker_trips_per_session: int = Field(
        default=2,
        ge=1,
        description="Batas maksimal kejadian turn loop breaker tripped per sesi sebelum memicu breach"
    )


# ============================================================================
# 3. GuardianReport Contract (Pydantic v2 Frozen)
# ============================================================================

class GuardianReport(BaseModel):
    """
    Snapshot terstruktur keadaan tata kelola satu sesi pada saat laporan eskalasi dibuat.
    Deep-frozen dan immutable.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(..., description="Identitas sesi agen percakapan")
    final_action: Literal["FORCE_REPORT", "TERMINATE"] = Field(
        ...,
        description="Tingkat eskalasi sesi saat laporan dibuat"
    )
    rejection_counts: Tuple[Tuple[str, int], ...] = Field(
        default=(),
        description="Pasangan (tool_name, jumlah_penolakan) terurut alfabetis (hanya tool dengan hitungan > 0)"
    )
    circuit_trips: int = Field(
        default=0,
        ge=0,
        description="Total frekuensi kejadian CIRCUIT_OPENED pada sesi"
    )
    turn_breaker_trips: int = Field(
        default=0,
        ge=0,
        description="Total frekuensi kejadian TURN_BREAKER_TRIPPED pada sesi"
    )
    opened_signatures: Tuple[str, ...] = Field(
        default=(),
        description="Daftar tanda tangan tool yang membuka circuit (urutan pertama, tanpa duplikat)"
    )
    event_sequence: Tuple[str, ...] = Field(
        default=(),
        description="Kronologi kind seluruh event pada sesi sesuai urutan kedatangan"
    )
    total_events: int = Field(
        default=0,
        ge=0,
        description="Total keseluruhan event yang tercatat pada jurnal sesi"
    )


# ============================================================================
# 4. GuardianDecision Contract (Pydantic v2 Frozen)
# ============================================================================

class GuardianDecision(BaseModel):
    """
    Keputusan tindakan tata kelola hasil evaluasi event oleh SmartGuardian.
    Deep-frozen dan immutable.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["CONTINUE", "WARN", "FORCE_REPORT", "TERMINATE"] = Field(
        ...,
        description="Tindakan yang direkomendasikan guardian bagi sesi saat ini"
    )
    reasons: Tuple[str, ...] = Field(
        default=(),
        description="Alasan eskalasi dalam Bahasa Indonesia baku (berdasarkan penghitung internal)"
    )
    report: Optional[GuardianReport] = Field(
        default=None,
        description="Snapshot laporan lengkap (WAJIB terisi bila FORCE_REPORT/TERMINATE, WAJIB None bila CONTINUE/WARN)"
    )

    @field_validator("reasons", mode="before")
    @classmethod
    def validate_reasons(cls, v: Any) -> Tuple[str, ...]:
        if v is None:
            return ()
        if isinstance(v, (list, tuple, set)):
            return tuple(str(x) for x in v)
        return (str(v),)

    @model_validator(mode="after")
    def validate_report_presence(self) -> "GuardianDecision":
        if self.action in ("FORCE_REPORT", "TERMINATE") and self.report is None:
            raise ValueError(
                f"GuardianDecision dengan action '{self.action}' WAJIB menyertakan 'report' (GuardianReport)."
            )
        if self.action in ("CONTINUE", "WARN") and self.report is not None:
            raise ValueError(
                f"GuardianDecision dengan action '{self.action}' WAJIB memiliki 'report'=None."
            )
        return self


# ============================================================================
# 5. Internal Session State Record
# ============================================================================

@dataclass
class _SessionState:
    """Keadaan privat dan mutable untuk melacak dinamika satu sesi percakapan."""
    journal: List[GovernanceEvent] = field(default_factory=list)
    rejection_counts: Dict[str, int] = field(default_factory=dict)
    circuit_trips: int = 0
    turn_breaker_trips: int = 0
    action_level: Literal["CONTINUE", "WARN", "FORCE_REPORT", "TERMINATE"] = "CONTINUE"
    opened_signatures: List[str] = field(default_factory=list)
    last_activity: float = 0.0


# ============================================================================
# 6. SmartGuardian Engine (Stateful In-Memory Aggregator)
# ============================================================================

class SmartGuardian:
    """
    Agregator sinyal tata kelola dan penilai eskalasi sesi berbasis kebijakan.
    
    Invarian:
    - Thread-safe: Seluruh manipulasi state dilindungi threading.Lock.
    - Monotonik: Level sesi tidak pernah turun (CONTINUE -> WARN -> FORCE_REPORT -> TERMINATE).
    - TERMINATE bersifat lengket: Sekali mencapai TERMINATE, seluruh keputusan berikutnya tetap TERMINATE.
    - Zero External Side-Effects: Tidak melakukan network, tidak menghentikan runtime secara paksa.
    """

    def __init__(self, policy: Optional[GovernancePolicy] = None) -> None:
        self.policy: GovernancePolicy = policy if policy is not None else GovernancePolicy()
        self._lock: threading.Lock = threading.Lock()
        self._sessions: Dict[str, _SessionState] = {}

    def _get_or_create_session(self, session_id: str, now: float) -> _SessionState:
        sess = self._sessions.get(session_id)
        if sess is None:
            sess = _SessionState(last_activity=now)
            self._sessions[session_id] = sess
        sess.last_activity = now
        return sess

    def _build_report(self, sess: _SessionState, session_id: str) -> GuardianReport:
        """Operasi baca murni untuk membangun snapshot laporan sesi terkini."""
        sorted_rejections = tuple(
            sorted(
                ((tool, count) for tool, count in sess.rejection_counts.items() if count > 0),
                key=lambda x: x[0],
            )
        )
        event_kinds = tuple(e.kind for e in sess.journal)

        # Level laporan hanya FORCE_REPORT atau TERMINATE
        report_level: Literal["FORCE_REPORT", "TERMINATE"] = (
            "TERMINATE" if sess.action_level == "TERMINATE" else "FORCE_REPORT"
        )

        return GuardianReport(
            session_id=session_id,
            final_action=report_level,
            rejection_counts=sorted_rejections,
            circuit_trips=sess.circuit_trips,
            turn_breaker_trips=sess.turn_breaker_trips,
            opened_signatures=tuple(sess.opened_signatures),
            event_sequence=event_kinds,
            total_events=len(sess.journal),
        )

    def observe_event(self, event: GovernanceEvent) -> GuardianDecision:
        """
        Mencatat event ke jurnal sesi dan mengevaluasi status eskalasi sesi.
        """
        now = time.monotonic()
        session_id = event.session_id

        with self._lock:
            sess = self._get_or_create_session(session_id, now)

            # 1. Catat ke jurnal
            sess.journal.append(event)

            # 2. Perbarui penghitung sesuai kind dan evaluasi apakah event ini memicu breach/warn
            event_is_breach = False
            event_is_warn = False
            breach_reasons: List[str] = []
            warn_reasons: List[str] = []

            if event.kind == "VERIFICATION_REJECTED":
                if event.tool_name:
                    count = sess.rejection_counts.get(event.tool_name, 0) + 1
                    sess.rejection_counts[event.tool_name] = count
                    if count >= self.policy.max_rejections_per_tool:
                        event_is_breach = True
                        breach_reasons.append(
                            f"Penolakan verifikasi CoV untuk tool '{event.tool_name}' mencapai ambang batas "
                            f"({count}/{self.policy.max_rejections_per_tool})."
                        )
                    elif count == (self.policy.max_rejections_per_tool - 1):
                        event_is_warn = True
                        warn_reasons.append(
                            f"Peringatan: Penolakan verifikasi CoV untuk tool '{event.tool_name}' mendekati ambang batas "
                            f"({count}/{self.policy.max_rejections_per_tool})."
                        )
            elif event.kind == "REJECTION_LIMIT_REACHED":
                event_is_breach = True
                if event.tool_name:
                    breach_reasons.append(
                        f"Sinyal REJECTION_LIMIT_REACHED diterima dari verifier untuk tool '{event.tool_name}'."
                    )
                else:
                    breach_reasons.append(
                        "Sinyal REJECTION_LIMIT_REACHED diterima dari verifier untuk sesi umum."
                    )
            elif event.kind == "CIRCUIT_OPENED":
                sess.circuit_trips += 1
                sig = event.detail.strip()
                if sig and sig not in sess.opened_signatures:
                    sess.opened_signatures.append(sig)

                if sess.circuit_trips >= self.policy.max_circuit_trips_per_session:
                    event_is_breach = True
                    breach_reasons.append(
                        f"Frekuensi circuit breaker terbuka mencapai ambang batas "
                        f"({sess.circuit_trips}/{self.policy.max_circuit_trips_per_session})."
                    )
                elif sess.circuit_trips == (self.policy.max_circuit_trips_per_session - 1):
                    event_is_warn = True
                    warn_reasons.append(
                        f"Peringatan: Frekuensi circuit breaker terbuka mendekati ambang batas "
                        f"({sess.circuit_trips}/{self.policy.max_circuit_trips_per_session})."
                    )
            elif event.kind == "TURN_BREAKER_TRIPPED":
                sess.turn_breaker_trips += 1
                if sess.turn_breaker_trips >= self.policy.max_turn_breaker_trips_per_session:
                    event_is_breach = True
                    breach_reasons.append(
                        f"Frekuensi turn loop breaker terpicu mencapai ambang batas "
                        f"({sess.turn_breaker_trips}/{self.policy.max_turn_breaker_trips_per_session})."
                    )
                elif sess.turn_breaker_trips == (self.policy.max_turn_breaker_trips_per_session - 1):
                    event_is_warn = True
                    warn_reasons.append(
                        f"Peringatan: Frekuensi turn loop breaker terpicu mendekati ambang batas "
                        f"({sess.turn_breaker_trips}/{self.policy.max_turn_breaker_trips_per_session})."
                    )

            # 3. Transisi Level Sesi (Monoton Naik)
            old_level = sess.action_level
            reasons_output: List[str] = []

            if old_level == "TERMINATE":
                new_level = "TERMINATE"
                reasons_output = breach_reasons if breach_reasons else [
                    "Sesi telah dihentikan secara permanen (TERMINATE). Event baru ditolak."
                ]
            elif event_is_breach:
                if old_level == "FORCE_REPORT":
                    new_level = "TERMINATE"
                    reasons_output = [
                        "Pelanggaran ambang batas tata kelola berulang setelah laporan eskalasi pertama: status dinaikkan ke TERMINATE."
                    ] + breach_reasons
                else:
                    new_level = "FORCE_REPORT"
                    reasons_output = breach_reasons
            elif event_is_warn:
                if old_level == "CONTINUE":
                    new_level = "WARN"
                    reasons_output = warn_reasons
                else:
                    new_level = old_level
                    # Bila sesi sudah FORCE_REPORT, event warn tidak menurunkan level dan mempertahankan alasan eskalasi
                    reasons_output = warn_reasons if old_level == "WARN" else [
                        f"Sesi berada pada level {old_level}. Aktivitas mendekati ambang tambahan terdeteksi."
                    ]
            else:
                new_level = old_level
                if new_level == "FORCE_REPORT":
                    reasons_output = [
                        f"Sesi berada pada level {new_level}. Melanjutkan pemantauan ketat."
                    ]

            sess.action_level = new_level

            # 4. Bangun Decision dan Report
            report_obj: Optional[GuardianReport] = None
            if new_level in ("FORCE_REPORT", "TERMINATE"):
                report_obj = self._build_report(sess, session_id)
                final_reasons = tuple(reasons_output) if reasons_output else (
                    f"Eskalasi sesi '{session_id}' berada pada level {new_level}.",
                )
            elif new_level == "WARN":
                final_reasons = tuple(reasons_output) if reasons_output else (
                    f"Peringatan: Aktivitas sesi '{session_id}' mendekati ambang batas governance.",
                )
            else:
                final_reasons = ()

            if new_level == "FORCE_REPORT" and old_level != "FORCE_REPORT":
                logger.warning(
                    f"[SmartGuardian] Sesi '{session_id}' ESKALASI ke FORCE_REPORT: {final_reasons}"
                )
            elif new_level == "TERMINATE" and old_level != "TERMINATE":
                logger.error(
                    f"[SmartGuardian] Sesi '{session_id}' ESKALASI KRITIS ke TERMINATE: {final_reasons}"
                )

            return GuardianDecision(
                action=new_level,
                reasons=final_reasons,
                report=report_obj,
            )

    def session_journal(self, session_id: str) -> Tuple[GovernanceEvent, ...]:
        """
        Mengembalikan jurnal event sesi sesuai urutan kedatangan.
        Sesi asing mengembalikan tuple kosong () tanpa efek samping.
        """
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return ()
            return tuple(sess.journal)

    def get_session_action(self, session_id: str) -> str:
        """
        Mengembalikan level eskalasi sesi saat ini ('CONTINUE' untuk sesi asing)
        tanpa mencatat event dan tanpa menciptakan entri baru.
        """
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return "CONTINUE"
            return sess.action_level

    def reset_session(self, session_id: str) -> None:
        """Menghapus seluruh state dan histori sesi. Idempoten dan aman untuk sesi asing."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def prune_idle_sessions(self, max_idle_seconds: float = 86400.0) -> int:
        """
        Menghapus seluruh record dari sesi yang aktivitas terakhirnya melampaui max_idle_seconds.
        Mengembalikan jumlah sesi yang dihapus.
        """
        now = time.monotonic()
        with self._lock:
            idle_sessions = [
                s_id for s_id, s_data in self._sessions.items()
                if (now - s_data.last_activity) > max_idle_seconds
            ]
            for s_id in idle_sessions:
                del self._sessions[s_id]
            pruned_count = len(idle_sessions)

        if pruned_count > 0:
            logger.debug(f"[SmartGuardian] Berhasil memangkas {pruned_count} sesi idle.")
        return pruned_count

    def clear_all_sessions(self) -> None:
        """Mengosongkan seluruh keadaan untuk isolasi pengujian."""
        with self._lock:
            self._sessions.clear()


# ============================================================================
# 7. Default Global Instance & Module-Level Delegation API
# ============================================================================

_DEFAULT_SMART_GUARDIAN = SmartGuardian()


def observe_event(event: GovernanceEvent) -> GuardianDecision:
    """Mendelegasikan pencatatan dan evaluasi event ke instance global default."""
    return _DEFAULT_SMART_GUARDIAN.observe_event(event)


def session_journal(session_id: str) -> Tuple[GovernanceEvent, ...]:
    """Mendelegasikan pembacaan jurnal sesi ke instance global default."""
    return _DEFAULT_SMART_GUARDIAN.session_journal(session_id)


def get_session_action(session_id: str) -> str:
    """Mendelegasikan pembacaan level eskalasi sesi ke instance global default."""
    return _DEFAULT_SMART_GUARDIAN.get_session_action(session_id)


def reset_session(session_id: str) -> None:
    """Mendelegasikan reset sesi ke instance global default."""
    _DEFAULT_SMART_GUARDIAN.reset_session(session_id)


def prune_idle_sessions(max_idle_seconds: float = 86400.0) -> int:
    """Mendelegasikan pemangkasan sesi idle ke instance global default."""
    return _DEFAULT_SMART_GUARDIAN.prune_idle_sessions(max_idle_seconds)


def clear_all_sessions() -> None:
    """Mendelegasikan pembersihan seluruh sesi ke instance global default."""
    _DEFAULT_SMART_GUARDIAN.clear_all_sessions()


__all__ = [
    "GovernanceEvent",
    "GovernancePolicy",
    "GuardianReport",
    "GuardianDecision",
    "SmartGuardian",
    "observe_event",
    "session_journal",
    "get_session_action",
    "reset_session",
    "prune_idle_sessions",
    "clear_all_sessions",
]
