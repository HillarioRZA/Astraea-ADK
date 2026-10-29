"""
Chain-of-Verification (CoV) Argument Verifier for LegionADK Governance.

Memeriksa keamanan argumen tool SEBELUM handler dieksekusi.
Fokus murni ADK (deterministik di RAM, tanpa panggilan model/jaringan eksternal).

Menyediakan:
1. VerificationVerdict: Kontrak keputusan verifikasi (Pydantic v2 frozen).
2. ToolVerifier: Protokol verifier argumen tool pra-eksekusi.
3. DeterministicVerifier: Implementasi verifier bawaan ADK berbasis aturan deterministik
   (path traversal, command injection, byte nul, ukuran maksimum string, dan circuit escalation).
"""

from __future__ import annotations

import logging
import os
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

# pyrefly: ignore[missing-import]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from LegionADK.core.llm_spec import LLMSpec
    from LegionADK.tools.permission_registry import ToolSpec

logger = logging.getLogger("LegionADK.governance.cov_verifier")

# Pola deteksi injeksi/pemisah perintah shell:
# Mengandung pipa ('|'), ampersand ('&'), backtick, newline ('\n', '\r'), subshell ('$('),
# atau titik koma (';') yang diikuti karakter perintah
COMMAND_INJECTION_PATTERN = re.compile(r"([|&`\n\r]|\$\(|;\s*\S+)")


# ============================================================================
# 1. VerificationVerdict Contract (Pydantic v2 Frozen)
# ============================================================================

class VerificationVerdict(BaseModel):
    """
    Kontrak keputusan hasil verifikasi argumen tool pra-eksekusi.
    Deep-frozen dan immutable untuk menjamin integritas tata kelola (governance).
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["APPROVED", "REJECTED"]
    tool_name: str = Field(..., description="Nama tool kanonis lowercase")
    risk_tier: Literal[0, 1, 2] = Field(..., description="Tingkat risiko tool yang diverifikasi")
    reasons: Tuple[str, ...] = Field(
        default=(),
        description="Daftar alasan penolakan dalam Bahasa Indonesia (wajib kosong bila APPROVED)"
    )
    verifier_id: str = Field(..., description="Identifikasi unik kelas/engine verifier")
    deterministic_only: bool = Field(
        ...,
        description="True bila verifikasi hanya mengandalkan aturan deterministik tanpa model semantik"
    )
    limit_reached: bool = Field(
        default=False,
        description="True bila penolakan ini mencapai batas toleransi penolakan berturut-turut"
    )

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("reasons", mode="before")
    @classmethod
    def validate_reasons(cls, v: Any) -> Tuple[str, ...]:
        if v is None:
            return ()
        if isinstance(v, (list, tuple, set)):
            return tuple(str(x) for x in v)
        return (str(v),)

    @model_validator(mode="after")
    def validate_status_reasons(self) -> "VerificationVerdict":
        if self.status == "APPROVED" and len(self.reasons) > 0:
            raise ValueError(
                f"VerificationVerdict dengan status 'APPROVED' tidak boleh memiliki reasons, "
                f"didapat {len(self.reasons)} alasan: {self.reasons}."
            )
        return self


# ============================================================================
# 2. ToolVerifier Protocol
# ============================================================================

class ToolVerifier(Protocol):
    """Protokol verifier argumen tool pra-eksekusi (Chain-of-Verification)."""

    async def verify(
        self,
        spec: LLMSpec,
        tool: ToolSpec,
        arguments: Dict[str, Any],
        context_messages: Sequence[Mapping[str, Any]],
    ) -> VerificationVerdict:
        """
        Memeriksa keamanan argumen pemanggilan tool sebelum dieksekusi oleh handler.
        Wajib bersifat murni membaca (read-only, tidak memutasi arguments milik pemanggil).
        """
        ...


# ============================================================================
# 3. DeterministicVerifier (Aturan Deterministik Bawaan ADK)
# ============================================================================

def _extract_string_leafs(data: Any, parent_key: str = "") -> List[Tuple[str, str]]:
    """
    Menelusuri struktur argumen (dict, list, tuple) secara rekursif pada kedalaman berapa pun
    dan mengumpulkan pasangan (nama_argumen_terdekat, nilai_string).
    """
    results: List[Tuple[str, str]] = []
    if isinstance(data, str):
        results.append((parent_key, data))
    elif isinstance(data, dict):
        for k, v in data.items():
            results.extend(_extract_string_leafs(v, str(k)))
    elif isinstance(data, (list, tuple, set)):
        for item in data:
            results.extend(_extract_string_leafs(item, parent_key))
    return results


class DeterministicVerifier:
    """
    Kelas verifier bawaan ADK berbasis aturan deterministik.
    Memeriksa argumen tool di memori tanpa panggilan jaringan maupun model critic LLM.
    
    Invarian:
    - Tidak pernah mengubah/memutasi dictionary arguments pemanggil (Invarian 8, non-hijacking).
    - Tier 0 selalu lolos (APPROVED) segera tanpa pemeriksaan dan tanpa menyentuh counter penolakan.
    - Tier 1 dan 2 diperiksa secara deterministik seragam.
    - String pada argumen non-path dan non-command (misal: 'content', 'query', 'prompt')
      TIDAK diperiksa traversal maupun injeksi shell.
    """
    verifier_id: str = "deterministic"

    def __init__(
        self,
        allowed_root: Optional[str] = None,
        max_string_bytes: Optional[int] = None,
        path_argument_names: Tuple[str, ...] = (
            "path", "file_path", "filepath", "filename", "file",
            "directory", "dir", "output_path", "target_path"
        ),
        command_argument_names: Tuple[str, ...] = (
            "command", "cmd", "script", "shell", "exec"
        ),
        max_consecutive_rejections: int = 3,
        on_escalation: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.allowed_root: Optional[str] = (
            os.path.abspath(allowed_root) if allowed_root else None
        )
        if max_string_bytes is None:
            # Lazy import di dalam konstruktor untuk memutus rantai circular dependency modul
            from LegionADK.tools.output_spill_guard import DEFAULT_MAX_OUTPUT_BYTES
            self.max_string_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
        else:
            self.max_string_bytes = max_string_bytes

        self.path_argument_names: Tuple[str, ...] = tuple(path_argument_names)
        self.command_argument_names: Tuple[str, ...] = tuple(command_argument_names)
        self.max_consecutive_rejections: int = max_consecutive_rejections
        self.on_escalation: Optional[Callable[[str, str], None]] = on_escalation

        # Set lowercase untuk lookup O(1)
        self._path_names_set = {n.strip().lower() for n in self.path_argument_names}
        self._cmd_names_set = {n.strip().lower() for n in self.command_argument_names}

        # Pelacakan penolakan berturut-turut di memori: (session_id, tool_name) -> int
        self._rejection_counts: Dict[Tuple[str, str], int] = {}

    def get_consecutive_rejections(self, session_id: str, tool_name: str) -> int:
        """Mengambil jumlah penolakan berturut-turut saat ini untuk pasangan (session_id, tool_name)."""
        clean_tool = tool_name.strip().lower()
        return self._rejection_counts.get((session_id, clean_tool), 0)

    def reset_rejection_count(self, session_id: str, tool_name: str) -> None:
        """Mereset penghitung penolakan untuk pasangan (session_id, tool_name)."""
        clean_tool = tool_name.strip().lower()
        self._rejection_counts.pop((session_id, clean_tool), None)

    def clear_all_rejections(self) -> None:
        """Mengosongkan seluruh histori penolakan di memori."""
        self._rejection_counts.clear()

    async def verify(
        self,
        spec: LLMSpec,
        tool: ToolSpec,
        arguments: Dict[str, Any],
        context_messages: Sequence[Mapping[str, Any]],
    ) -> VerificationVerdict:
        """
        Alur verifikasi deterministik berurutan:
        1. Jika tool.risk_tier == 0: kembalikan APPROVED segera.
        2. Kumpulkan semua nilai string rekursif dari arguments.
        3. Terapkan aturan:
           - Semua string: periksa byte nul (\x00) dan panjang maksimum bytes.
           - Argumen bernama path: periksa segmen path traversal ('..') dan batas allowed_root.
           - Argumen bernama perintah: periksa pola injeksi/pemisah shell ('|', '&', ';', newline, backtick, '$(').
           - Argumen lain (content, query, dll): tidak diperiksa traversal/injeksi.
        4. Jika tidak ada pelanggaran: kembalikan APPROVED dan reset counter penolakan.
        5. Jika ada pelanggaran: kembalikan REJECTED, naikkan counter, dan evaluasi limit_reached / escalation.
        """
        canonical_tool_name = tool.name.strip().lower()
        session_id = spec.handle

        # 1. Tool Tier 0 selalu APPROVED segera tanpa pemeriksaan & tanpa menyentuh counter
        if tool.risk_tier == 0:
            return VerificationVerdict(
                status="APPROVED",
                tool_name=canonical_tool_name,
                risk_tier=0,
                reasons=(),
                verifier_id=self.verifier_id,
                deterministic_only=True,
                limit_reached=False,
            )

        # 2. Kumpulkan semua nilai string secara rekursif
        string_leafs = _extract_string_leafs(arguments)
        reasons: List[str] = []

        # 3. Evaluasi aturan per string leaf
        for arg_name, val in string_leafs:
            clean_arg_name = arg_name.strip()
            arg_name_lower = clean_arg_name.lower()

            # Aturan A: Semua string, nama apa pun
            # - Mengandung byte nul (\x00)
            if "\x00" in val:
                reasons.append(
                    f"Argumen '{clean_arg_name}' ditolak: mengandung karakter byte nul (\\x00)."
                )

            # - Panjang byte UTF-8 melebihi max_string_bytes
            val_bytes_len = len(val.encode("utf-8"))
            if val_bytes_len > self.max_string_bytes:
                reasons.append(
                    f"Argumen '{clean_arg_name}' ditolak: panjang ukuran ({val_bytes_len} bytes) "
                    f"melampaui batas maksimum ({self.max_string_bytes} bytes)."
                )

            # Aturan B: Hanya argumen bernama path
            if arg_name_lower in self._path_names_set:
                # Periksa pola path traversal: pecah segmen berdasarkan '/' dan '\'
                # Menangkap '..', 'sub/..', '..\\dir', dan '../path'
                normalized_path = val.replace("\\", "/")
                segments = [seg.strip() for seg in normalized_path.split("/") if seg.strip()]
                if ".." in segments:
                    reasons.append(
                        f"Argumen '{clean_arg_name}' ditolak: mengandung pola path traversal '..' pada segmen path."
                    )

                # Periksa path absolut terhadap allowed_root
                if os.path.isabs(val) and self.allowed_root:
                    norm_val = os.path.abspath(val)
                    norm_root = os.path.abspath(self.allowed_root)
                    try:
                        common = os.path.commonpath([norm_val, norm_root])
                        if common != norm_root:
                            reasons.append(
                                f"Argumen '{clean_arg_name}' ditolak: path absolut berada di luar direktori yang diizinkan (allowed_root)."
                            )
                    except ValueError:
                        # Terjadi pada Windows jika lintas drive letter (misal C: vs E:)
                        reasons.append(
                            f"Argumen '{clean_arg_name}' ditolak: path absolut berada di luar direktori yang diizinkan (allowed_root)."
                        )

            # Aturan C: Hanya argumen bernama perintah
            if arg_name_lower in self._cmd_names_set:
                if COMMAND_INJECTION_PATTERN.search(val):
                    reasons.append(
                        f"Argumen '{clean_arg_name}' ditolak: mengandung karakter/operator pemisah perintah shell "
                        f"('|', '&', ';', newline, backtick, atau '$(')."
                    )

        tracking_key = (session_id, canonical_tool_name)

        # 4. Bila tidak ada pelanggaran: kembalikan APPROVED dan reset counter
        if not reasons:
            self._rejection_counts.pop(tracking_key, None)
            return VerificationVerdict(
                status="APPROVED",
                tool_name=canonical_tool_name,
                risk_tier=tool.risk_tier,
                reasons=(),
                verifier_id=self.verifier_id,
                deterministic_only=True,
                limit_reached=False,
            )

        # 5. Bila ada pelanggaran: kembalikan REJECTED, naikkan counter, dan evaluasi eskalasi
        current_rejections = self._rejection_counts.get(tracking_key, 0) + 1
        self._rejection_counts[tracking_key] = current_rejections
        limit_reached = current_rejections >= self.max_consecutive_rejections

        # Callback on_escalation hanya dipanggil tepat satu kali saat batas penolakan pertama kali tercapai
        is_first_escalation = current_rejections == self.max_consecutive_rejections
        if is_first_escalation:
            logger.warning(
                f"[CoVVerifier] Batas penolakan berturut-turut tercapai untuk session '{session_id}' "
                f"pada tool '{canonical_tool_name}' ({current_rejections}/{self.max_consecutive_rejections})."
            )
            if self.on_escalation is not None:
                try:
                    self.on_escalation(session_id, canonical_tool_name)
                except Exception as exc:
                    logger.error(
                        f"[CoVVerifier] Gagal mengeksekusi callback on_escalation untuk '{session_id}': {exc}",
                        exc_info=True,
                    )

        return VerificationVerdict(
            status="REJECTED",
            tool_name=canonical_tool_name,
            risk_tier=tool.risk_tier,
            reasons=tuple(reasons),
            verifier_id=self.verifier_id,
            deterministic_only=True,
            limit_reached=limit_reached,
        )


__all__ = [
    "VerificationVerdict",
    "ToolVerifier",
    "DeterministicVerifier",
]
