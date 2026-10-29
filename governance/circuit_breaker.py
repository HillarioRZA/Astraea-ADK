"""
Tool-Signature Circuit Breaker Engine for LegionADK Governance.

Melacak dan memutus loop kegagalan pemanggilan tool yang identik (nama tool + argumen sama persis)
dalam satu sesi percakapan.

Perbedaan Arsitektur terhadap Breaker Level Turn:
====================================================================================================
| Aspek                | Turn-Level Breaker (turn_compaction.py) | Tool-Signature Breaker (circuit_breaker.py)   |
|----------------------|-----------------------------------------|-----------------------------------------------|
| Cakupan Deteksi      | Putaran percakapan LLM tanpa kemajuan   | Eksekusi tool dengan aksi & argumen identik   |
| Unit Pelacakan       | Turn berturut-turut tanpa artefak baru  | Pasangan (session_id, sha256(tool + args))    |
| Pemicu Trip          | Ambang batas loop putaran kosong        | Eksekusi tool identik gagal >= threshold kali |
| Reaksi Sistem        | Intervensi turn compaction / stop turn  | Status OPEN: blokir pemanggilan tool identik  |
| Peran Modul          | Algoritma kontrol konteks percakapan    | Pencatat status murni & state machine trip    |
====================================================================================================

Catatan Desain:
- Modul ini bertindak murni sebagai PENCATAT status eksekusi yang dilaporkan oleh pemanggil.
- Modul ini tidak mengeksekusi tool dan tidak melempar error untuk kegagalan eksekusi.
- Penolakan oleh CoV Verifier secara eksplisit BUKAN kegagalan eksekusi dan tidak dilaporkan ke modul ini.
- Zero Internal Dependencies: Tidak mengimpor modul LegionADK lain dalam bentuk apa pun.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import threading
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

# pyrefly: ignore[missing-import]
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("LegionADK.governance.circuit_breaker")


# ============================================================================
# 1. CircuitState Contract (Pydantic v2 Frozen)
# ============================================================================

class CircuitState(BaseModel):
    """
    Kontrak status sirkuit untuk satu tanda tangan tool dalam sesi tertentu.
    Deep-frozen dan immutable untuk mencegah manipulasi state dari luar breaker.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["CLOSED", "OPEN"] = Field(
        ...,
        description="Status sirkuit: CLOSED (normal/dapat dipanggil) atau OPEN (terputus/diblokir)"
    )
    consecutive_failures: int = Field(
        ...,
        ge=0,
        description="Jumlah kegagalan eksekusi berturut-turut yang tercatat saat ini"
    )
    threshold: int = Field(
        default=3,
        ge=1,
        description="Ambang batas toleransi kegagalan berturut-turut sebelum sirkuit trip ke OPEN"
    )
    signature: str = Field(
        ...,
        description="Hash SHA-256 kanonis dari nama tool dan argumen pemanggilan"
    )


# ============================================================================
# 2. Tool Signature Computation (Pure Function)
# ============================================================================

def _normalize_arguments_recursive(obj: Any) -> Any:
    """
    Menormalisasi struktur data argumen secara rekursif sebelum serialisasi JSON:
    - Mapping -> dict dengan kunci string (str(k)).
    - list dan tuple -> list (keduanya setara dalam representasi argumen tool).
    - set dan frozenset -> list yang diurutkan berdasarkan serialisasi kanonisnya.
    - Nilai primitif lainnya dibiarkan apa adanya.
    """
    if isinstance(obj, Mapping):
        return {str(k): _normalize_arguments_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_normalize_arguments_recursive(x) for x in obj]
    elif isinstance(obj, (set, frozenset)):
        norm_list = [_normalize_arguments_recursive(x) for x in obj]
        return sorted(norm_list, key=lambda x: json.dumps(x, sort_keys=True, default=str))
    return obj


def compute_tool_signature(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """
    Menghitung digest SHA-256 kanonis deterministik untuk pasangan (tool_name, arguments).

    Aturan Normalisasi:
    1. Nama tool dinormalisasi dengan .strip().lower() (selaras dengan kanonisasi registry).
    2. Argumen dinormalisasi secara rekursif:
       - Mapping dikonversi menjadi dict dengan kunci bertipe string.
       - Tuple dan list diperlakukan setara sebagai list.
       - Set dan frozenset diurutkan deterministik.
    3. Payload diserialisasi dengan json.dumps(sort_keys=True, default=str).

    Keputusan Desain Sadar (Known Limitation):
    Angka bulat `1` dan pecahan `1.0` menghasilkan tanda tangan yang BERBEDA karena serialisasi
    JSON menghasilkan token string berbeda ('1' vs '1.0'). Normalisasi numerik (float-to-int)
    tidak dilakukan pada versi ini demi menjaga presisi data asli.

    Invarian:
    Fungsi ini murni (pure function): tidak membaca/menulis state breaker apa pun dan
    tidak mengubah/memutasi objek arguments milik pemanggil.
    """
    canonical_tool = tool_name.strip().lower()
    normalized_args = _normalize_arguments_recursive(arguments)

    payload = json.dumps(
        {"tool": canonical_tool, "arguments": normalized_args},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ============================================================================
# 3. Internal Mutable Record
# ============================================================================

@dataclass
class _BreakerRecord:
    """Record internal privat untuk melacak kegagalan per tanda tangan."""
    consecutive_failures: int = 0
    last_activity: float = 0.0


# ============================================================================
# 4. ToolCircuitBreaker Class (Thread-Safe State Machine)
# ============================================================================

class ToolCircuitBreaker:
    """
    Circuit Breaker level tanda tangan tool untuk mendeteksi dan menghentikan pengulangan
    kegagalan aksi yang sama persis dalam suatu sesi percakapan.

    Thread-safety:
    Seluruh operasi baca/tulis terhadap state internal dilindungi oleh threading.Lock.
    Callback `on_trip` selalu dipanggil SETELAH lock dilepas untuk mencegah deadlock.
    """

    def __init__(
        self,
        threshold: int = 3,
        on_trip: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        if threshold < 1:
            raise ValueError(f"Threshold circuit breaker harus >= 1, didapat: {threshold}")
        self.threshold: int = threshold
        self.on_trip: Optional[Callable[[str, str], None]] = on_trip

        self._lock: threading.Lock = threading.Lock()
        # Key: (session_id, signature) -> _BreakerRecord
        self._records: Dict[Tuple[str, str], _BreakerRecord] = {}
        # Key: session_id -> timestamp aktivitas terakhir sesi (monotonic)
        self._session_activity: Dict[str, float] = {}

    def record_tool_outcome(
        self,
        session_id: str,
        signature: str,
        succeeded: bool,
    ) -> CircuitState:
        """
        Mencatat hasil eksekusi tool (sukses atau gagal) untuk tanda tangan tertentu.

        Perilaku:
        - succeeded=True: Mereset hitungan kegagalan menjadi 0 dan status menjadi CLOSED.
        - succeeded=False: Menaikkan hitungan kegagalan. Jika mencapai threshold dan sebelumnya
          CLOSED, status berubah menjadi OPEN dan callback on_trip dipanggil. Kegagalan berikutnya
          saat sudah OPEN tidak memanggil on_trip lagi.
        """
        now = time.monotonic()
        key = (session_id, signature)
        must_trip = False

        with self._lock:
            self._session_activity[session_id] = now
            record = self._records.get(key)
            if record is None:
                record = _BreakerRecord(consecutive_failures=0, last_activity=now)
                self._records[key] = record

            record.last_activity = now

            if succeeded:
                record.consecutive_failures = 0
                state_status: Literal["CLOSED", "OPEN"] = "CLOSED"
            else:
                was_closed = record.consecutive_failures < self.threshold
                record.consecutive_failures += 1
                is_now_open = record.consecutive_failures >= self.threshold

                if was_closed and is_now_open:
                    must_trip = True

                state_status = "OPEN" if is_now_open else "CLOSED"

            failures_count = record.consecutive_failures

        # Callback on_trip dipanggil di luar lock untuk mencegah potensi deadlock
        if must_trip:
            logger.warning(
                f"[ToolCircuitBreaker] Circuit TRIP ke status OPEN untuk session '{session_id}', "
                f"signature '{signature}' (kegagalan berturut-turut: {failures_count}/{self.threshold})."
            )
            if self.on_trip is not None:
                try:
                    self.on_trip(session_id, signature)
                except Exception as exc:
                    logger.error(
                        f"[ToolCircuitBreaker] Gagal mengeksekusi callback on_trip untuk session '{session_id}' "
                        f"signature '{signature}': {exc}",
                        exc_info=True,
                    )

        return CircuitState(
            status=state_status,
            consecutive_failures=failures_count,
            threshold=self.threshold,
            signature=signature,
        )

    def get_circuit_state(self, session_id: str, signature: str) -> CircuitState:
        """
        Membaca status sirkuit saat ini untuk pasangan (session_id, signature).
        Bila belum pernah tercatat, mengembalikan status CLOSED dengan kegagalan 0
        TANPA membuat entri baru pada memori internal.
        """
        key = (session_id, signature)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return CircuitState(
                    status="CLOSED",
                    consecutive_failures=0,
                    threshold=self.threshold,
                    signature=signature,
                )

            status: Literal["CLOSED", "OPEN"] = (
                "OPEN" if record.consecutive_failures >= self.threshold else "CLOSED"
            )
            return CircuitState(
                status=status,
                consecutive_failures=record.consecutive_failures,
                threshold=self.threshold,
                signature=signature,
            )

    def is_open(self, session_id: str, signature: str) -> bool:
        """Helper ringkas untuk mengecek apakah sirkuit saat ini dalam kondisi OPEN."""
        return self.get_circuit_state(session_id, signature).status == "OPEN"

    def reset_session(self, session_id: str) -> None:
        """
        Menghapus seluruh record dan histori status sirkuit untuk satu sesi tertentu.
        Operasi ini idempoten dan aman bila sesi tidak ditemukan.
        """
        with self._lock:
            keys_to_delete = [k for k in self._records if k[0] == session_id]
            for k in keys_to_delete:
                del self._records[k]
            self._session_activity.pop(session_id, None)

    def prune_idle_sessions(self, max_idle_seconds: float = 86400.0) -> int:
        """
        Menghapus seluruh record dari sesi yang aktivitas terakhirnya melampaui max_idle_seconds.
        Mengembalikan jumlah sesi yang berhasil dibersihkan.
        """
        now = time.monotonic()
        with self._lock:
            idle_sessions = [
                sess for sess, last_active in self._session_activity.items()
                if (now - last_active) > max_idle_seconds
            ]

            for sess in idle_sessions:
                keys_to_delete = [k for k in self._records if k[0] == sess]
                for k in keys_to_delete:
                    del self._records[k]
                del self._session_activity[sess]

            pruned_count = len(idle_sessions)

        if pruned_count > 0:
            logger.debug(f"[ToolCircuitBreaker] Berhasil memangkas {pruned_count} sesi idle.")
        return pruned_count

    def clear_all_sessions(self) -> None:
        """Mengosongkan seluruh histori dan record sirkuit (berguna untuk isolasi testing)."""
        with self._lock:
            self._records.clear()
            self._session_activity.clear()


# ============================================================================
# 5. Default Global Instance & Module-Level Delegation API
# ============================================================================

_DEFAULT_CIRCUIT_BREAKER = ToolCircuitBreaker(threshold=3)


def record_tool_outcome(
    session_id: str,
    signature: str,
    succeeded: bool,
) -> CircuitState:
    """Mendelegasikan pencatatan hasil eksekusi tool ke instance global default."""
    return _DEFAULT_CIRCUIT_BREAKER.record_tool_outcome(
        session_id=session_id,
        signature=signature,
        succeeded=succeeded,
    )


def get_circuit_state(session_id: str, signature: str) -> CircuitState:
    """Mendelegasikan pembacaan status sirkuit ke instance global default."""
    return _DEFAULT_CIRCUIT_BREAKER.get_circuit_state(
        session_id=session_id,
        signature=signature,
    )


def is_open(session_id: str, signature: str) -> bool:
    """Mendelegasikan pengecekan status OPEN ke instance global default."""
    return _DEFAULT_CIRCUIT_BREAKER.is_open(
        session_id=session_id,
        signature=signature,
    )


def reset_session(session_id: str) -> None:
    """Mendelegasikan reset sesi ke instance global default."""
    _DEFAULT_CIRCUIT_BREAKER.reset_session(session_id)


def prune_idle_sessions(max_idle_seconds: float = 86400.0) -> int:
    """Mendelegasikan pemangkasan sesi idle ke instance global default."""
    return _DEFAULT_CIRCUIT_BREAKER.prune_idle_sessions(max_idle_seconds)


def clear_all_sessions() -> None:
    """Mendelegasikan pembersihan seluruh sesi ke instance global default."""
    _DEFAULT_CIRCUIT_BREAKER.clear_all_sessions()


__all__ = [
    "CircuitState",
    "compute_tool_signature",
    "ToolCircuitBreaker",
    "record_tool_outcome",
    "get_circuit_state",
    "is_open",
    "reset_session",
    "prune_idle_sessions",
    "clear_all_sessions",
]
