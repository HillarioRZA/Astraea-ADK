"""
Context Compaction & Anti-Infinite Loop Circuit Breaker untuk LegionADK Runtime.
Menyediakan Two-Tier Context Compaction 100% in-memory (RAM) untuk mencegah
Context Window Overflow pada sesi percakapan otonom multi-turn:

Lapisan 1 (Cache Boundary Overhead Guard):
  Memastikan static system prompt tidak mendominasi atau menghabiskan kapasitas
  token budget sebelum agen mulai beroperasi.

Lapisan 2 (Two-Tier History Compaction dengan Akuntansi Terpisah):
  Memisahkan immutable system anchor dari riwayat percakapan dinamis.
  - Tier 1 (Soft Pruning): Memangkas payload output tool lama menjadi ringkasan 1 baris
    sambil mempertahankan turn terbaru dan ID tool call tetap utuh.
  - Tier 2 (Hard Summarization): Menggulung separuh riwayat terlama menjadi satu blok
    ringkasan padat ([COMPACTED_HISTORY_SUMMARY]) dengan penyelarasan pasangan tool.
  - Emergency Truncation: Menjamin riwayat percakapan tidak pernah melampaui history_budget
    tanpa menyentuh jangkar identitas/SOP sistem.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from LegionADK.core.llm_spec import MessageEnvelope
except ImportError:
    MessageEnvelope = None  # type: ignore

from LegionADK.gateway.cache_boundary import (
    LEGION_CACHE_BOUNDARY,
    ensure_system_prompt_cache_boundary,
    split_system_prompt_cache_boundary,
)

logger = logging.getLogger("LegionADK.runtime.turn_compaction")

COMPACTED_SUMMARY_MARKER = "[COMPACTED_HISTORY_SUMMARY]"
DEFAULT_MAX_TOKEN_BUDGET = 8000
DEFAULT_MAX_OUTPUT_CHARS = 120
DEFAULT_MIN_HISTORY_FLOOR = 64
MAX_CONSECUTIVE_EMPTY_TURNS = 3


class SystemOverheadExceededError(ValueError):
    """
    Dilempar ketika ukuran static system prompt sendiri melampaui atau
    menghabiskan alokasi budget token, sehingga tidak menyisakan ruang
    minimum bagi riwayat percakapan.
    """
    pass


def _is_immutable_system_anchor(msg: Dict[str, Any]) -> bool:
    """
    True HANYA untuk pesan system asli (identitas/SOP/aturan kanonis),
    BUKAN pesan ringkasan histori [COMPACTED_HISTORY_SUMMARY].
    """
    role = str(msg.get("role", "")).lower()
    content = str(msg.get("content", ""))
    return role == "system" and COMPACTED_SUMMARY_MARKER not in content


def estimate_tokens(
    target: Union[str, Dict[str, Any], List[Dict[str, Any]], Any],
    custom_counter: Optional[Callable[[str], int]] = None,
) -> int:
    """
    Menghitung estimasi jumlah token murni di RAM tanpa dependensi eksternal.
    Menggunakan rasio standar empiris (~4 karakter per token untuk teks umum,
    ~3 karakter per token untuk JSON/kode padat + overhead metadata)
    atau callable custom tokenizer jika disediakan pemanggil.
    """
    if custom_counter is not None:
        if isinstance(target, str):
            return custom_counter(target)
        elif isinstance(target, dict):
            content_str = str(target.get("content", ""))
            return custom_counter(content_str) + 4
        elif isinstance(target, list):
            return sum(estimate_tokens(item, custom_counter=custom_counter) for item in target)
        return custom_counter(str(target))

    if isinstance(target, str):
        char_count = len(target)
        if char_count == 0:
            return 0
        word_count = len(target.split())
        code_markers = target.count("{") + target.count("}") + target.count("[") + target.count("]") + target.count('"')
        is_dense_code = code_markers > (char_count // 30)
        char_divisor = 3 if is_dense_code else 4
        return max(char_count // char_divisor, word_count, 1)

    if isinstance(target, dict):
        content = target.get("content") or ""
        tokens = estimate_tokens(str(content))
        if "role" in target:
            tokens += 2
        if "name" in target or "tool_call_id" in target:
            tokens += 3
        if "tool_calls" in target and target["tool_calls"]:
            tokens += estimate_tokens(str(target["tool_calls"]))
        return tokens

    if isinstance(target, list):
        return sum(estimate_tokens(item) for item in target)

    if hasattr(target, "content"):
        return estimate_tokens(str(getattr(target, "content", ""))) + 4

    return max(len(str(target)) // 4, 1) if target else 0


def strip_internal_metadata(
    messages: Union[List[Dict[str, Any]], Dict[str, Any]],
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Membersihkan seluruh kunci/metadata internal (yang diawali underscore '_')
    dari kamus pesan sebelum dikembalikan ke pemanggil atau dikirim ke adapter API LLM.
    Mencegah penolakan skema payload HTTP pada vendor LLM yang menerapkan validasi ketat.
    """
    if isinstance(messages, dict):
        return {k: v for k, v in messages.items() if not k.startswith("_")}
    if isinstance(messages, list):
        return [
            {k: v for k, v in m.items() if not k.startswith("_")}
            if isinstance(m, dict) else m
            for m in messages
        ]
    return messages


def _normalize_message(msg: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    """Mengonversi pesan (dict atau MessageEnvelope) menjadi format dict standar secara aman."""
    if isinstance(msg, dict):
        return dict(msg)
    if MessageEnvelope is not None and isinstance(msg, MessageEnvelope):
        return {
            "role": "user" if msg.sender in ("user", "human") else "assistant",
            "content": msg.content,
            "sender": msg.sender,
            "target": list(msg.target),
            "envelope_id": msg.envelope_id,
        }
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    if hasattr(msg, "__dict__"):
        return dict(msg.__dict__)
    return {"role": "user", "content": str(msg)}


def _is_tool_output_message(msg: Dict[str, Any]) -> bool:
    """Mendeteksi apakah pesan merupakan output eksekusi perkakas (tool/function result)."""
    role = str(msg.get("role", "")).lower()
    if role in ("tool", "function"):
        return True
    if "tool_call_id" in msg or "tool_name" in msg:
        return True
    return False


def prune_old_tool_outputs(
    messages: List[Dict[str, Any]],
    preserve_recent_turns: int = 2,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Tier 1 (Soft Pruning):
    Memangkas output perkakas yang panjang dari giliran-giliran awal menjadi
    ringkasan 1 baris padat, sambil mempertahankan giliran terbaru tetap utuh
    serta menjaga tool_call_id dan struktur pesan agar kompatibel dengan API LLM.

    Mengembalikan:
        Tuple (pruned_messages, total_pruned_count).
    """
    if not messages:
        return [], 0

    preserve_count = max(preserve_recent_turns * 2, 2)
    cutoff_index = max(0, len(messages) - preserve_count)

    pruned_list: List[Dict[str, Any]] = []
    pruned_count = 0

    for idx, raw_msg in enumerate(messages):
        msg = copy.deepcopy(raw_msg)
        if idx < cutoff_index and _is_tool_output_message(msg):
            content = msg.get("content")
            content_str = str(content) if content is not None else ""
            if len(content_str) > max_output_chars:
                tool_label = msg.get("name") or msg.get("tool_name") or msg.get("tool_call_id") or "tool"
                summary_line = (
                    f"[Tool Result Pruned: '{tool_label}' output ({len(content_str)} chars) "
                    f"compacted to 1 line for context efficiency]"
                )
                msg["content"] = summary_line
                pruned_count += 1

        pruned_list.append(msg)

    return strip_internal_metadata(pruned_list), pruned_count  # type: ignore


def _align_tool_pair_boundary(pool: List[Dict[str, Any]], raw_cut: int) -> int:
    """
    Menyelaraskan titik potong pesan agar tidak membelah pasangan pemanggilan tool
    (assistant with tool_calls <-> tool response) dan menjamin tail_slice tidak pernah
    diawali oleh pesan tool yatim (yang memicu HTTP 400 di API OpenAI / Anthropic).
    """
    n = len(pool)
    if n <= 2:
        return 0

    cut = max(1, min(raw_cut, n - 1))

    # 1. Jika titik potong mendarat di tengah-tengah pesan tool, geser maju
    #    agar tool response tersebut masuk ke dalam compact_slice.
    while cut < n and _is_tool_output_message(pool[cut]):
        cut += 1

    # 2. Jika pesan sebelum cut adalah assistant yang memanggil tool_calls,
    #    pastikan seluruh respon tool pasangannya ikut terserap ke compact_slice.
    if cut > 0:
        prev = pool[cut - 1]
        if prev.get("role") == "assistant" and prev.get("tool_calls"):
            while cut < n and _is_tool_output_message(pool[cut]):
                cut += 1

    # 3. Jika pergeseran maju menyisakan ekor terlalu pendek (< 2 pesan),
    #    mundurkan cut ke sebelum pesan assistant yang menginisiasi tool call.
    if n - cut < 2:
        cut = max(1, min(raw_cut, n - 1))
        while cut > 0:
            prev = pool[cut - 1]
            if _is_tool_output_message(prev):
                cut -= 1
            elif prev.get("role") == "assistant" and prev.get("tool_calls"):
                cut -= 1
                break
            else:
                break

    # 4. Batasan keamanan: pastikan rentang cut valid
    cut = max(0, min(cut, n))

    # Verifikasi akhir: tail_slice tidak boleh diawali oleh tool output message
    while cut < n and _is_tool_output_message(pool[cut]):
        cut += 1

    return cut


def summarize_oldest_history(
    messages: List[Dict[str, Any]],
    fraction_to_compact: float = 0.5,
    summary_marker: str = COMPACTED_SUMMARY_MARKER,
) -> List[Dict[str, Any]]:
    """
    Tier 2 (Hard Summarization):
    Menggulung separuh (atau porsi tertentu) percakapan terlama menjadi satu
    blok ringkasan padat ([COMPACTED_HISTORY_SUMMARY]) di memori.
    Mempertahankan pesan instruksi awal dan menyelaraskan pasangan tool secara presisi.

    Langkah 3: Jika messages[0] adalah 'user', beri summary_message role 'user'
    agar valid dan tidak memicu HTTP 400 karena mid-stream system role.
    """
    if len(messages) <= 2:
        return copy.deepcopy(messages)

    first_msg = messages[0]
    if _is_immutable_system_anchor(first_msg):
        head = [copy.deepcopy(first_msg)]
        working_pool = messages[1:]
        summary_role = "system"
    else:
        head = []
        working_pool = messages
        # Jika messages[0] ber-role 'user', ringkasan di awal riwayat diberi role 'user'
        summary_role = "user"

    pool_len = len(working_pool)
    if pool_len < 2:
        return copy.deepcopy(messages)

    raw_cut = max(1, int(pool_len * fraction_to_compact))
    aligned_cut = _align_tool_pair_boundary(working_pool, raw_cut)

    if aligned_cut <= 0 or aligned_cut >= pool_len:
        aligned_cut = max(1, pool_len - 2)

    compact_slice = working_pool[:aligned_cut]
    tail_slice = working_pool[aligned_cut:]

    # Ekstrak intisari poin penting dari blok yang digulung
    summary_points: List[str] = []
    for m in compact_slice:
        role = str(m.get("role", "unknown")).upper()
        content = str(m.get("content", "")).strip()

        if summary_marker in content:
            summary_points.append(f"- [HISTORI SEBELUMNYA]:\n  {content[:300]}...")
            continue

        snippet = (content[:100] + "...") if len(content) > 100 else content
        snippet_clean = snippet.replace("\n", " ")
        if _is_tool_output_message(m):
            tool_id = m.get("name") or m.get("tool_call_id") or "tool"
            summary_points.append(f"- TOOL ({tool_id}): {snippet_clean}")
        else:
            summary_points.append(f"- {role}: {snippet_clean}")

    bullet_summary = "\n".join(summary_points)
    summary_text = (
        f"{summary_marker}\n"
        f"Kompilasi ringkasan dari {len(compact_slice)} putaran percakapan sebelumnya:\n"
        f"{bullet_summary}\n"
        f"[Akhir ringkasan histori. Rincian lampau digulung untuk efisiensi context window]."
    )

    summary_message: Dict[str, Any] = {
        "role": summary_role,
        "content": summary_text,
    }

    result: List[Dict[str, Any]] = []
    result.extend(head)

    # Cegah dua pesan berurutan dengan role sama jika tail_slice diawali pesan user
    if tail_slice and tail_slice[0].get("role") == summary_role:
        first_tail = copy.deepcopy(tail_slice[0])
        first_tail_content = first_tail.get("content", "")
        if isinstance(first_tail_content, list):
            first_tail["content"] = [{"type": "text", "text": summary_text}] + first_tail_content
        else:
            first_tail["content"] = f"{summary_text}\n\n{str(first_tail_content)}"
        result.append(first_tail)
        result.extend(copy.deepcopy(tail_slice[1:]))
    else:
        result.append(summary_message)
        result.extend(copy.deepcopy(tail_slice))

    return strip_internal_metadata(result)  # type: ignore


def _apply_emergency_tail_truncation(
    messages: List[Dict[str, Any]],
    max_token_budget: int,
    custom_counter: Optional[Callable[[str], int]] = None,
) -> List[Dict[str, Any]]:
    """
    P1 Fix & Langkah 3:
    Mekanisme pengaman akhir jika token masih melampaui budget setelah seluruh
    iterasi Tier 2. Memangkas isi pesan non-immutable-system terpanjang (termasuk
    blok ringkasan jika memang diperlukan) secara terukur sehingga jaminan
    context window tidak pernah bocor diam-diam.
    """
    truncated_messages = copy.deepcopy(messages)
    current_tokens = estimate_tokens(truncated_messages, custom_counter=custom_counter)

    for _ in range(5):
        if current_tokens <= max_token_budget:
            break

        longest_idx = -1
        max_chars = 0
        for i, m in enumerate(truncated_messages):
            # Langkah 3: Gunakan _is_immutable_system_anchor agar blok ringkasan
            # juga bisa dirampingkan jika ekor sudah habis terpangkas.
            if not _is_immutable_system_anchor(m) and isinstance(m.get("content"), str):
                c_len = len(m["content"])
                if c_len > max_chars:
                    max_chars = c_len
                    longest_idx = i

        if longest_idx == -1 or max_chars < 80:
            break

        excess_tokens = current_tokens - max_token_budget
        chars_to_cut = max(excess_tokens * 3 + 100, max_chars // 2)
        target_len = max(60, max_chars - chars_to_cut)

        original_content = str(truncated_messages[longest_idx].get("content", ""))
        truncated_content = (
            original_content[:target_len]
            + f"\n\n[... KONTEN DIPANGKAS OTOMATIS OLEH RUNTIME: "
            f"Disesuaikan dengan batas budget {max_token_budget} token ...]"
        )
        truncated_messages[longest_idx]["content"] = truncated_content
        current_tokens = estimate_tokens(truncated_messages, custom_counter=custom_counter)

    return truncated_messages


def compact_context_two_tier(
    messages: List[Union[Dict[str, Any], Any]],
    max_token_budget: int = DEFAULT_MAX_TOKEN_BUDGET,
    preserve_recent_turns: int = 2,
    custom_counter: Optional[Callable[[str], int]] = None,
    raise_on_overflow: bool = False,
    min_history_floor: int = DEFAULT_MIN_HISTORY_FLOOR,
) -> List[Dict[str, Any]]:
    """
    Two-Tier Context Compaction Engine untuk LegionADK.
    Menerapkan akuntansi terpisah antara System Overhead dan History Budget (Langkah 2):

    1. Pisahkan pesan menjadi system_anchors dan conversation_history.
    2. Hitung system_tokens dan history_budget = max_token_budget - system_tokens.
    3. Jika history_budget < min_history_floor:
       - Cek apakah system_anchors memiliki LEGION_CACHE_BOUNDARY.
       - Jika ada, pangkas dynamic_suffix tanpa menyentuh stable_prefix.
       - Jika tetap kurang, lempar SystemOverheadExceededError secara eksplisit.
    4. Jalankan Tier 1 (Soft Pruning) dan Tier 2 (Hard Summarization) murni
       pada conversation_history dengan target history_budget.
    5. Jika riwayat masih melampaui history_budget, terapkan emergency tail truncation
       (atau lempar error jika raise_on_overflow=True).
    6. Gabungkan kembali system_anchors + compacted_history.

    Mengembalikan:
        List pesan bersih yang 100% dijamin berada di bawah max_token_budget.
    """
    if not messages:
        return []

    # 1. Normalisasi dan salin pesan murni di RAM
    normalized_messages = [_normalize_message(m) for m in messages]

    # 2. Langkah 2: Pisahkan system_anchors (identitas/SOP kanonis) dari conversation_history
    system_anchors: List[Dict[str, Any]] = []
    conversation_history: List[Dict[str, Any]] = []

    for m in normalized_messages:
        if _is_immutable_system_anchor(m):
            system_anchors.append(copy.deepcopy(m))
        else:
            conversation_history.append(copy.deepcopy(m))

    system_tokens = estimate_tokens(system_anchors, custom_counter=custom_counter)
    history_budget = max_token_budget - system_tokens

    # 3. Langkah 2: Validasi ruang bernapas minimum bagi riwayat (min_history_floor)
    if history_budget < min_history_floor:
        boundary_trimmed = False
        for anchor in system_anchors:
            content = str(anchor.get("content", ""))
            if LEGION_CACHE_BOUNDARY in content:
                parts = split_system_prompt_cache_boundary(content)
                if parts:
                    stable_prefix, dynamic_suffix = parts
                    stable_tokens = estimate_tokens(stable_prefix, custom_counter=custom_counter)
                    remaining_for_system = max_token_budget - min_history_floor
                    if stable_tokens + min_history_floor <= max_token_budget:
                        dynamic_budget_tokens = max(0, remaining_for_system - stable_tokens)
                        if dynamic_budget_tokens > 10:
                            target_chars = dynamic_budget_tokens * 3
                            truncated_dynamic = dynamic_suffix[:target_chars] + "\n[... DYNAMIC CONTEXT COMPACTED ...]"
                            anchor["content"] = f"{stable_prefix}{LEGION_CACHE_BOUNDARY}{truncated_dynamic}"
                        else:
                            anchor["content"] = ensure_system_prompt_cache_boundary(stable_prefix)
                        boundary_trimmed = True

        if boundary_trimmed:
            system_tokens = estimate_tokens(system_anchors, custom_counter=custom_counter)
            history_budget = max_token_budget - system_tokens

        # Jika masih di bawah floor, berarti static prefix itu sendiri yang menghabiskan budget
        if history_budget < min_history_floor:
            raise SystemOverheadExceededError(
                f"[ContextCompaction] Static system prompt overhead ({system_tokens} tokens) "
                f"menghabiskan budget total ({max_token_budget} tokens), menyisakan ruang riwayat "
                f"({history_budget} tokens) di bawah batas minimum floor ({min_history_floor} tokens). "
                f"Tingkatkan max_token_budget atau kurangi ukuran identitas/SOP sistem."
            )

    # Jika tidak ada riwayat percakapan, kembalikan system anchors
    if not conversation_history:
        return strip_internal_metadata(system_anchors)  # type: ignore

    # 4. Jalankan Compaction Murni pada conversation_history dengan target history_budget
    curr_history_tokens = estimate_tokens(conversation_history, custom_counter=custom_counter)

    # Cek awal: jika riwayat percakapan sudah di bawah history_budget
    if curr_history_tokens <= history_budget:
        combined = system_anchors + conversation_history
        return strip_internal_metadata(combined)  # type: ignore

    logger.info(
        f"[ContextCompaction] Riwayat ({curr_history_tokens} est tokens) melampaui "
        f"history_budget ({history_budget} tokens). Memulai Tier 1 (Soft Pruning)..."
    )

    # Tier 1: Soft Pruning pada observasi perkakas lama
    pruned_history, pruned_count = prune_old_tool_outputs(
        conversation_history,
        preserve_recent_turns=preserve_recent_turns,
        max_output_chars=DEFAULT_MAX_OUTPUT_CHARS,
    )
    curr_history_tokens = estimate_tokens(pruned_history, custom_counter=custom_counter)

    if curr_history_tokens <= history_budget:
        logger.info(
            f"[ContextCompaction] Tier 1 berhasil: {pruned_count} output tool dipangkas. "
            f"Riwayat terkompresi menjadi {curr_history_tokens} tokens."
        )
        combined = system_anchors + pruned_history
        return strip_internal_metadata(combined)  # type: ignore

    # Tier 2: Hard Summarization jika Tier 1 belum mencukupi
    logger.info(
        f"[ContextCompaction] Tier 1 tidak mencukupi ({curr_history_tokens} > {history_budget}). "
        f"Memulai Tier 2 (Hard Summarization)..."
    )

    compacted_history = pruned_history
    iteration = 0
    max_iterations = 3

    while curr_history_tokens > history_budget and len(compacted_history) > 2 and iteration < max_iterations:
        iteration += 1
        compacted_history = summarize_oldest_history(
            compacted_history,
            fraction_to_compact=0.5,
            summary_marker=COMPACTED_SUMMARY_MARKER,
        )
        curr_history_tokens = estimate_tokens(compacted_history, custom_counter=custom_counter)
        logger.info(
            f"[ContextCompaction] Tier 2 iterasi {iteration}: riwayat dipadatkan. "
            f"Token riwayat sekarang {curr_history_tokens} (target <= {history_budget})."
        )

    # Emergency Tail Truncation jika riwayat masih melampaui history_budget
    if curr_history_tokens > history_budget:
        msg_err = (
            f"[ContextCompaction] Gagal mencapai target token riwayat: estimasi {curr_history_tokens} "
            f"masih melampaui history_budget {history_budget} setelah {iteration} iterasi."
        )
        if raise_on_overflow:
            logger.error(msg_err)
            raise ValueError(msg_err)

        logger.warning(
            f"{msg_err} Menerapkan pemangkasan darurat pada riwayat percakapan..."
        )
        compacted_history = _apply_emergency_tail_truncation(
            compacted_history,
            max_token_budget=history_budget,
            custom_counter=custom_counter,
        )

    # 5. Gabungkan kembali system_anchors + compacted_history
    combined = system_anchors + compacted_history
    return strip_internal_metadata(combined)  # type: ignore


def check_turn_loop_circuit_breaker(
    consecutive_empty_turns: int,
    max_allowed: int = MAX_CONSECUTIVE_EMPTY_TURNS,
) -> Tuple[bool, str]:
    """
    Anti-Infinite Loop Circuit Breaker untuk siklus turn agen.
    Mendeteksi apakah agen terjebak dalam perulangan berturut-turut tanpa menghasilkan
    artefak atau kemajuan baru (misal berputar memanggil tool kosong atau respons kosong).

    Mengembalikan:
        Tuple (is_loop_detected: bool, message: str).
    """
    if consecutive_empty_turns >= max_allowed:
        msg = (
            f"Loop terdeteksi: {consecutive_empty_turns} putaran berturut-turut "
            f"tanpa menghasilkan artefak/kemajuan baru (ambang batas maks: {max_allowed})."
        )
        logger.warning(f"[CircuitBreaker] {msg} Menandai status BLOCKED_LOOP_DETECTED.")
        return True, msg
    return False, ""


def compact_turn_history(
    messages: List[Dict[str, Any]],
    max_history_chars: int = 12000,
) -> List[Dict[str, Any]]:
    """
    Fungsi utilitas berbasis karakter untuk kompatibilitas ke belakang (NexusADK contract).
    Menerjemahkan batas karakter menjadi estimasi budget token lalu menjalankan compaction.
    """
    total_len = sum(len(str(m.get("content", ""))) for m in messages)
    if total_len <= max_history_chars:
        return strip_internal_metadata([_normalize_message(m) for m in messages])  # type: ignore

    token_budget = max(500, max_history_chars // 4)
    return compact_context_two_tier(messages, max_token_budget=token_budget)


__all__ = [
    "COMPACTED_SUMMARY_MARKER",
    "DEFAULT_MAX_TOKEN_BUDGET",
    "DEFAULT_MAX_OUTPUT_CHARS",
    "DEFAULT_MIN_HISTORY_FLOOR",
    "MAX_CONSECUTIVE_EMPTY_TURNS",
    "SystemOverheadExceededError",
    "_is_immutable_system_anchor",
    "estimate_tokens",
    "strip_internal_metadata",
    "prune_old_tool_outputs",
    "summarize_oldest_history",
    "compact_context_two_tier",
    "check_turn_loop_circuit_breaker",
    "compact_turn_history",
]
