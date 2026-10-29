"""
Fase 10 Turn Engine (Finalizer) untuk LegionADK.

Fungsi murni di akhir giliran untuk merakit TurnResult dari potongan-potongan giliran
(riwayat request, hasil panggilan model, dan hasil putaran tool) serta memperbarui
SessionState secara deterministik tanpa I/O atau mutasi objek masukan.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from LegionADK.runtime.turn_state import TurnResult
from LegionADK.runtime.turn_tool_round import build_tool_messages

if TYPE_CHECKING:
    from LegionADK.core.llm_spec import MessageEnvelope
    from LegionADK.governance.smart_guardian import GuardianReport
    from LegionADK.runtime.turn_api import ApiOutcome
    from LegionADK.runtime.turn_state import SessionState
    from LegionADK.runtime.turn_tool_round import ToolRoundOutcome


def finalize_turn(
    *,
    initial_state: SessionState,
    envelope: MessageEnvelope,
    request_messages: Sequence[Dict[str, Any]],
    rounds: Sequence[Tuple[ApiOutcome, Optional[ToolRoundOutcome]]],
    status: str,
    message: str = "",
    guardian_report: Optional[GuardianReport] = None,
) -> TurnResult:
    """
    Merakit TurnResult akhir dan memperbarui SessionState dari seluruh putaran giliran.

    Fungsi ini bersifat murni (tanpa I/O dan tanpa efek samping pada objek masukan).
    Status akhir ditegakkan oleh konstruktor TurnResult.
    """
    # 1. Riwayat baru
    new_history: List[Dict[str, Any]] = list(request_messages)
    for api_outcome, round_outcome in rounds:
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": api_outcome.content,
        }
        if api_outcome.tool_calls:
            assistant_msg["tool_calls"] = [dict(tc) for tc in api_outcome.tool_calls]
        new_history.append(assistant_msg)

        if round_outcome is not None:
            new_history.extend(build_tool_messages(round_outcome))

    # 2. Metrik giliran
    total_prompt = sum(ao.prompt_tokens for ao, _ in rounds)
    total_cached = sum(ao.cached_tokens for ao, _ in rounds)
    total_completion = sum(ao.completion_tokens for ao, _ in rounds)

    # 3. Hasil eksekusi tool gabungan
    all_tool_results: List[Dict[str, Any]] = []
    for _, ro in rounds:
        if ro is not None:
            all_tool_results.extend(ro.results)
    tool_results: Tuple[Dict[str, Any], ...] = tuple(all_tool_results)

    # 4. Evaluasi kekosongan giliran
    has_progress = any(bool(ao.tool_calls) or bool(ao.content and ao.content.strip()) for ao, _ in rounds)
    empty = not has_progress

    # 5. Bangun SessionState baru
    new_state = initial_state.with_history(new_history).record_turn(
        prompt_tokens=total_prompt,
        cached_tokens=total_cached,
        completion_tokens=total_completion,
        empty=empty,
    )

    # 6. Konten dan penalaran akhir dari putaran terakhir
    if rounds:
        last_api_outcome = rounds[-1][0]
        final_content = last_api_outcome.content
        final_reasoning = last_api_outcome.reasoning
    else:
        final_content = ""
        final_reasoning = ""

    # 7. Kembalikan TurnResult
    return TurnResult(
        status=status,
        content=final_content,
        reasoning=final_reasoning,
        tool_results=tool_results,
        prompt_tokens=total_prompt,
        cached_tokens=total_cached,
        completion_tokens=total_completion,
        rounds_used=len(rounds),
        session_state=new_state,
        guardian_report=guardian_report,
        message=message,
    )


__all__ = ["finalize_turn"]
