"""
Fase 5-8 Turn Engine (Putaran Tool) untuk LegionADK.

Mengeksekusi seluruh pemanggilan tool dari satu putaran model secara berurutan,
mengomposisikan pipeline governance lengkap:
- Repair argumen mentah melalui gateway/tool_call_repair.
- Tool-Signature Circuit Breaker untuk mencegah loop kegagalan identik.
- RBAC dan registri permission via tools/permission_registry.
- Chain-of-Verification (CoV) pra-eksekusi via governance/cov_verifier.
- Smart Guardian event observation dan eskalasi via governance/smart_guardian.
- Eksekusi handler terdaftar dengan spill guard 50 KB.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

# pyrefly: ignore[missing-import]
from pydantic import BaseModel, ConfigDict, Field

from LegionADK.gateway.tool_call_repair import repair_malformed_tool_call
from LegionADK.governance.circuit_breaker import (
    ToolCircuitBreaker,
    compute_tool_signature,
)
from LegionADK.governance.cov_verifier import DeterministicVerifier
from LegionADK.governance.smart_guardian import GovernanceEvent, GuardianReport
from LegionADK.tools.permission_registry import (
    ToolArgumentError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionError,
    authorize_tool_call,
    execute_registered_tool,
    get_registered_tool,
)

if TYPE_CHECKING:
    from LegionADK.core.llm_spec import CancellationToken, LLMSpec
    from LegionADK.governance.smart_guardian import SmartGuardian

logger = logging.getLogger("LegionADK.runtime.turn_tool_round")


class ToolRoundOutcome(BaseModel):
    """
    Kontrak hasil eksekusi satu putaran pemanggilan tool (Fase 5-8).
    Frozen dan immutable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    results: Tuple[Dict[str, Any], ...] = Field(
        default=(),
        description="Daftar hasil eksekusi setiap pemanggilan tool sesuai urutan panggilan",
    )
    repair_needed: bool = Field(
        default=False,
        description="Menandakan apakah ada tool call yang ditolak oleh verifikasi CoV",
    )
    repair_feedback: Optional[str] = Field(
        default=None,
        description="Umpan balik perbaikan dari penolakan CoV terakhir dalam putaran ini",
    )
    tool_calls_requested: int = Field(
        default=0,
        ge=0,
        description="Jumlah total pemanggilan tool yang diminta dalam putaran ini",
    )
    guardian_action: Optional[str] = Field(
        default=None,
        description="Keputusan tindakan terakhir dari Smart Guardian yang diamati selama putaran ini",
    )
    guardian_report: Optional[GuardianReport] = Field(
        default=None,
        description="Laporan eskalasi Smart Guardian terbaru jika dihasilkan selama putaran ini",
    )


def build_tool_messages(outcome: ToolRoundOutcome) -> List[Dict[str, Any]]:
    """
    Membangun daftar pesan riwayat percakapan bertipe tool dari hasil putaran tool.
    Fungsi sinkron dan murni: satu pesan per hasil dengan urutan yang sama persis.
    """
    messages: List[Dict[str, Any]] = []
    for res in outcome.results:
        tool_call_id = res.get("tool_call_id", "")
        status = res.get("status")
        if status == "success":
            content = res.get("output", "")
        elif status == "rejected_by_cov":
            content = res.get("feedback", "")
        else:
            content = res.get("error", "")

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(content) if content is not None else "",
        })
    return messages


async def execute_tool_round(
    spec: LLMSpec,
    session_id: str,
    raw_tool_calls: Sequence[Dict[str, Any]],
    context_messages: Sequence[Dict[str, Any]],
    *,
    verifier: Optional[DeterministicVerifier] = None,
    guardian: Optional[SmartGuardian] = None,
    signature_breaker: Optional[ToolCircuitBreaker] = None,
    token: Optional[CancellationToken] = None,
) -> ToolRoundOutcome:
    """
    Mengeksekusi satu putaran pemanggilan perkakas (tool round) secara berurutan.
    Menjalankan alur:
    0. Repair sekali di depan untuk menambal malformed tool call syntax.
    1. Pemeriksaan circuit breaker berbasis tanda tangan.
    2. Validasi hak akses RBAC spec terhadap nama tool.
    3. Verifikasi ketersediaan spesifikasi tool di registry.
    4. Verifikasi deterministik CoV (jika verifier disediakan) dan pelaporan event ke Guardian.
    5. Eksekusi registered tool handler (dengan spill guard) dan pencatatan hasil ke breaker.
    """
    # 0. Repair sekali di depan
    repaired = repair_malformed_tool_call(list(raw_tool_calls))
    tool_calls_requested = len(repaired)

    results: List[Dict[str, Any]] = []
    last_guardian_action: Optional[str] = None
    last_guardian_report: Optional[GuardianReport] = None

    for entry in repaired:
        name = ""
        arguments: Any = None
        tool_call_id = ""

        if isinstance(entry, dict):
            name = entry.get("name", "")
            arguments = entry.get("arguments")
            tool_call_id = entry.get("id", "")

            # Tangani format function call bersarang (OpenAI style) bila name kosong
            if not name and "function" in entry and isinstance(entry["function"], dict):
                name = entry["function"].get("name", "")
                if arguments is None:
                    arguments = entry["function"].get("arguments")

            # Fallback ke tool_id/parameters bila format custom
            if not name:
                name = entry.get("tool_id", "") or entry.get("tool", "")
            if arguments is None and "parameters" in entry:
                arguments = entry.get("parameters")

        # Parsing string arguments JSON hasil repair bila ada
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                pass

        if not isinstance(arguments, dict):
            arguments = {} if arguments is None else arguments

        # 1. Tanda tangan dan Breaker
        signature = compute_tool_signature(name, arguments)
        if signature_breaker is not None and signature_breaker.is_open(session_id, signature):
            results.append({
                "tool_call_id": tool_call_id,
                "tool_name": name,
                "status": "circuit_open",
                "error": f"Pemanggilan tool '{name}' ditolak: circuit breaker terbuka untuk pola panggilan ini.",
            })
            continue

        # 2. RBAC Guard
        if not authorize_tool_call(spec, name):
            results.append({
                "tool_call_id": tool_call_id,
                "tool_name": name,
                "status": "permission_denied",
                "error": f"Tool '{name}' tidak diizinkan untuk agen ini.",
            })
            continue

        # 3. Registry Lookup
        tool_spec = get_registered_tool(name)
        if tool_spec is None:
            results.append({
                "tool_call_id": tool_call_id,
                "tool_name": name,
                "status": "not_found",
                "error": f"Tool '{name}' tidak terdaftar.",
            })
            continue

        # 4. Chain-of-Verification (CoV) Pra-Eksekusi
        if verifier is not None:
            verdict = await verifier.verify(spec, tool_spec, arguments, context_messages)
            if verdict.status == "REJECTED":
                feedback = "; ".join(verdict.reasons)
                results.append({
                    "tool_call_id": tool_call_id,
                    "tool_name": name,
                    "status": "rejected_by_cov",
                    "feedback": feedback,
                })

                if guardian is not None:
                    dec1 = guardian.observe_event(
                        GovernanceEvent(
                            kind="VERIFICATION_REJECTED",
                            session_id=session_id,
                            tool_name=name,
                        )
                    )
                    last_guardian_action = dec1.action
                    if dec1.report is not None:
                        last_guardian_report = dec1.report

                    if verdict.limit_reached:
                        dec2 = guardian.observe_event(
                            GovernanceEvent(
                                kind="REJECTION_LIMIT_REACHED",
                                session_id=session_id,
                                tool_name=name,
                            )
                        )
                        last_guardian_action = dec2.action
                        if dec2.report is not None:
                            last_guardian_report = dec2.report

                # Penolakan CoV TIDAK dicatat ke breaker tanda tangan dalam keadaan apa pun
                continue

        # 5. Eksekusi Registered Tool
        try:
            output = await execute_registered_tool(spec, name, arguments, token)
            results.append({
                "tool_call_id": tool_call_id,
                "tool_name": name,
                "status": "success",
                "output": output,
            })
            if signature_breaker is not None:
                signature_breaker.record_tool_outcome(session_id, signature, True)
        except ToolArgumentError as exc:
            results.append({
                "tool_call_id": tool_call_id,
                "tool_name": name,
                "status": "invalid_arguments",
                "error": str(exc),
            })
            if signature_breaker is not None:
                signature_breaker.record_tool_outcome(session_id, signature, False)
        except ToolExecutionError as exc:
            results.append({
                "tool_call_id": tool_call_id,
                "tool_name": name,
                "status": "execution_error",
                "error": str(exc),
            })
            if signature_breaker is not None:
                signature_breaker.record_tool_outcome(session_id, signature, False)
        except ToolPermissionError:
            results.append({
                "tool_call_id": tool_call_id,
                "tool_name": name,
                "status": "permission_denied",
                "error": f"Tool '{name}' tidak diizinkan untuk agen ini.",
            })
        except ToolNotFoundError:
            results.append({
                "tool_call_id": tool_call_id,
                "tool_name": name,
                "status": "not_found",
                "error": f"Tool '{name}' tidak terdaftar.",
            })
        except asyncio.CancelledError:
            raise

    # 6. Penutup Putaran
    has_cov_rejection = False
    latest_repair_feedback: Optional[str] = None

    for res in results:
        if res.get("status") == "rejected_by_cov":
            has_cov_rejection = True
            latest_repair_feedback = res.get("feedback")

    return ToolRoundOutcome(
        results=tuple(results),
        repair_needed=has_cov_rejection,
        repair_feedback=latest_repair_feedback,
        tool_calls_requested=tool_calls_requested,
        guardian_action=last_guardian_action,
        guardian_report=last_guardian_report,
    )


__all__ = [
    "ToolRoundOutcome",
    "build_tool_messages",
    "execute_tool_round",
]
