"""
Fase 1-10 Turn Engine Orchestrator untuk LegionADK.

Menyediakan fungsi utama `execute_turn` sebagai fasad orkestrasi seluruh fase giliran:
- Fase 1: Preflight check (pembatalan token, kuota max turns, governance session action).
- Fase 9a: Hard context compaction (Two-Tier RAM compaction).
- Fase 2: Perakitan payload TurnRequest lengkap (system cache boundary, skema tools, route config).
- Fase 3-8: Loop eksekusi pemanggilan model (invoke_llm) dan putaran eksekusi perkakas (execute_tool_round).
- Fase 9b: Anti-infinite loop circuit breaker guard & governance escalation.
- Fase 10: Finalisasi giliran (finalize_turn) dan pembaruan SessionState.
"""

from __future__ import annotations

import asyncio
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from LegionADK.gateway.cache_boundary import assemble_cached_system_prompt_from_spec
from LegionADK.gateway.model_router import ModelRouter
from LegionADK.governance.cov_verifier import DeterministicVerifier
from LegionADK.governance.smart_guardian import GovernanceEvent
from LegionADK.runtime.turn_api import TurnTimeoutError, invoke_llm
from LegionADK.runtime.turn_compaction import (
    SystemOverheadExceededError,
    check_turn_loop_circuit_breaker,
    compact_context_two_tier,
    estimate_tokens,
)
from LegionADK.runtime.turn_finalizer import finalize_turn
from LegionADK.runtime.turn_preflight import run_preflight
from LegionADK.runtime.turn_request import TurnRequest, assemble_request
from LegionADK.runtime.turn_state import TurnPolicy, TurnResult
from LegionADK.runtime.turn_tool_round import build_tool_messages, execute_tool_round

if TYPE_CHECKING:
    from LegionADK.core.llm_spec import CancellationToken, LLMSpec, MessageEnvelope
    from LegionADK.governance.circuit_breaker import ToolCircuitBreaker
    from LegionADK.governance.smart_guardian import GuardianReport, SmartGuardian
    from LegionADK.runtime.turn_api import ApiOutcome
    from LegionADK.runtime.turn_state import SessionState
    from LegionADK.runtime.turn_tool_round import ToolRoundOutcome


def _hasil_awal(state: SessionState, status: str, message: str) -> TurnResult:
    """Helper privat untuk membangun TurnResult pada jalur keluar awal (pre-round)."""
    return TurnResult(
        status=status,
        content="",
        reasoning="",
        tool_results=(),
        prompt_tokens=0,
        cached_tokens=0,
        completion_tokens=0,
        rounds_used=0,
        session_state=state,
        guardian_report=None,
        message=message,
    )


async def execute_turn(
    spec: LLMSpec,
    envelope: MessageEnvelope,
    session_state: SessionState,
    *,
    router: Optional[ModelRouter] = None,
    verifier: Optional[DeterministicVerifier] = None,
    guardian: Optional[SmartGuardian] = None,
    signature_breaker: Optional[ToolCircuitBreaker] = None,
    token: Optional[CancellationToken] = None,
    policy: Optional[TurnPolicy] = None,
    dynamic_context: str = "",
) -> TurnResult:
    """
    Mengeksekusi satu giliran percakapan agen secara otonom dari Fase 1 hingga Fase 10.
    """
    active_router = router if router is not None else ModelRouter()
    active_policy = policy if policy is not None else TurnPolicy()
    active_verifier = verifier if verifier is not None else DeterministicVerifier()

    # Fase 1: Preflight
    pre = run_preflight(session_state, active_policy, token=token, guardian=guardian)
    if not pre.can_proceed:
        return _hasil_awal(session_state, pre.status, pre.reason)

    # Fase 9a: Context Compaction & Budget Synchronization
    if active_policy.context_token_budget is None:
        history_for_request = list(session_state.history)
    else:
        try:
            # 1. Rakit system_prompt terlebih dahulu via cache_boundary
            # (sekaligus memvalidasi Lapisan 1 SystemOverheadExceededError)
            parts: List[str] = [f"Pengirim: {envelope.sender}"]
            if envelope.reply_to is not None:
                parts.append(f"Membalas pesan: {envelope.reply_to}")
            if envelope.attachments:
                parts.append(f"Lampiran: {', '.join(envelope.attachments)}")
            if dynamic_context.strip():
                parts.append(dynamic_context)
            assembled_context = "\n\n".join(parts)

            system_prompt, _, _ = assemble_cached_system_prompt_from_spec(
                spec,
                dynamic_context=assembled_context,
                max_token_budget=active_policy.context_token_budget,
            )

            # 2. Hitung overhead token (system prompt + konten pesan user envelope)
            overhead_tokens = estimate_tokens(system_prompt) + estimate_tokens(envelope.content)
            budget_for_history = active_policy.context_token_budget - overhead_tokens

            if budget_for_history < 0:
                raise SystemOverheadExceededError(
                    f"Overhead prompt ({overhead_tokens} tokens) melampaui context_token_budget "
                    f"({active_policy.context_token_budget} tokens)."
                )

            # 3. Jalankan compact_context_two_tier dengan sisa jatah budget untuk riwayat
            history_for_request = compact_context_two_tier(
                list(session_state.history),
                max_token_budget=budget_for_history,
            )
        except SystemOverheadExceededError as exc:
            return _hasil_awal(session_state, "BUDGET_EXCEEDED", str(exc))

    # Fase 2: Perakitan Request
    try:
        request = assemble_request(
            spec,
            envelope,
            history_for_request,
            dynamic_context=dynamic_context,
            context_token_budget=active_policy.context_token_budget,
        )
    except SystemOverheadExceededError as exc:
        return _hasil_awal(session_state, "BUDGET_EXCEEDED", str(exc))

    # Putaran (Fase 3-8)
    rounds: List[Tuple[ApiOutcome, Optional[ToolRoundOutcome]]] = []
    running: List[Dict[str, Any]] = list(request.messages)
    status = "COMPLETED"
    message = ""
    latest_report: Optional[GuardianReport] = None
    tool_rounds_done = 0

    while True:
        # a. Bangun request putaran ini
        if not rounds:
            req = request
        else:
            req = TurnRequest(
                system_prompt=request.system_prompt,
                system_digest=request.system_digest,
                tool_digest=request.tool_digest,
                tools=request.tools,
                messages=tuple(running),
                route=request.route,
            )

        # b. Panggil model LLM via invoke_llm
        try:
            outcome = await invoke_llm(
                active_router,
                req,
                token=token,
                timeout_seconds=active_policy.api_timeout_seconds,
            )
        except TurnTimeoutError as exc:
            status = "ERROR"
            message = str(exc)
            break
        except asyncio.CancelledError:
            status = "CANCELLED"
            if token is not None and token.is_cancelled and token.reason:
                message = f"Dibatalkan: {token.reason}"
            else:
                message = "Giliran dibatalkan."
            break
        except Exception as exc:
            status = "ERROR"
            message = f"Kegagalan router: {exc}"[:300]
            break

        # c. Bila model tidak memanggil tool apa pun
        if not outcome.tool_calls:
            rounds.append((outcome, None))
            break

        # d. Eksekusi putaran tool
        try:
            round_outcome = await execute_tool_round(
                spec,
                session_state.session_id,
                outcome.tool_calls,
                running,
                verifier=active_verifier,
                guardian=guardian,
                signature_breaker=signature_breaker,
                token=token,
            )
        except asyncio.CancelledError:
            status = "CANCELLED"
            if token is not None and token.is_cancelled and token.reason:
                message = f"Dibatalkan: {token.reason}"
            else:
                message = "Giliran dibatalkan."
            break

        # e. Catat putaran dan perbarui pesan berjalan
        rounds.append((outcome, round_outcome))
        tool_rounds_done += 1

        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": outcome.content,
            "tool_calls": [dict(tc) for tc in outcome.tool_calls],
        }
        running.append(assistant_msg)
        running.extend(build_tool_messages(round_outcome))

        # f. Evaluasi laporan dan eskalasi guardian
        if round_outcome.guardian_report is not None:
            latest_report = round_outcome.guardian_report
        if round_outcome.guardian_action == "TERMINATE":
            status = "TERMINATED"
            break

        # g. Periksa batas maksimal putaran tool
        if tool_rounds_done >= active_policy.max_tool_rounds:
            status = "IN_PROGRESS"
            if round_outcome.repair_needed and round_outcome.repair_feedback:
                message = round_outcome.repair_feedback
            else:
                message = "Batas putaran tool tercapai."
            break

    # Fase 9b: Loop Guard
    has_progress = any(bool(ao.tool_calls) or bool(ao.content and ao.content.strip()) for ao, _ in rounds)
    candidate = 0 if has_progress else session_state.consecutive_empty_turns + 1
    tripped, trip_message = check_turn_loop_circuit_breaker(
        candidate,
        max_allowed=active_policy.max_consecutive_empty_turns,
    )
    if tripped:
        if guardian is not None:
            decision = guardian.observe_event(
                GovernanceEvent(
                    kind="TURN_BREAKER_TRIPPED",
                    session_id=session_state.session_id,
                    detail=trip_message,
                )
            )
            if decision.action == "TERMINATE" and status != "TERMINATED":
                status = "TERMINATED"
                if decision.report is not None:
                    latest_report = decision.report

        if status in ("COMPLETED", "IN_PROGRESS"):
            status = "BLOCKED_LOOP"
            if not message:
                message = trip_message

    # Fase 10: Finalizer
    return finalize_turn(
        initial_state=session_state,
        envelope=envelope,
        request_messages=request.messages,
        rounds=rounds,
        status=status,
        message=message,
        guardian_report=latest_report,
    )


__all__ = ["execute_turn"]
