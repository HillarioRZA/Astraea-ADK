"""
Tool Call Syntax & JSON Repair Engine untuk LegionADK.
Mem-parsing, mengekstrak, dan menambal argumen JSON tool yang rusak atau terpotong oleh LLM
sebelum divalidasi oleh skema tool agen.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("LegionADK.gateway.tool_call_repair")


def _repair_common_json_errors(raw_json: str) -> str:
    """
    Melakukan perbaikan heuristik pada kesalahan umum sintaks JSON yang dihasilkan oleh LLM:
    - Menghilangkan trailing comma (koma menggantung) sebelum penutup kurung siku/kurawal atau di akhir string.
    - Mengonversi literal Python (True, False, None) ke (true, false, null) HANYA di luar tanda kutip string.
    - Mengonversi single quote Python/JS ke double quote JSON yang valid.
    - Menutup tanda kutip ganda yang terpotong.
    - Menyeimbangkan tanda kurung kurawal `{}` dan siku `[]` secara LIFO (stack-based) agar urutan penutupan valid.
    """
    s = raw_json.strip()

    # 0. Hapus blok markdown jika ada di dalam snippet
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()

    # 1. Ganti literal Python menjadi JSON standard HANYA di luar string literal
    def _replace_literals(match: re.Match) -> str:
        if match.group(1):  # Di dalam tanda kutip string ganda atau tunggal
            return match.group(1)
        lit = match.group(2)
        if lit == "True":
            return "true"
        elif lit == "False":
            return "false"
        elif lit == "None":
            return "null"
        return lit

    s = re.sub(
        r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|(\bTrue\b|\bFalse\b|\bNone\b)',
        _replace_literals,
        s,
    )

    # 2. Hapus trailing comma sebelum penutup kurung kurawal atau siku, atau di akhir string
    s = re.sub(r",\s*([\]}])", r"\1", s)
    s = re.sub(r",\s*$", "", s)

    # 3. Ganti single quote yang bertindak sebagai pembungkus kunci/nilai
    if "'" in s and '"' not in s:
        s = s.replace("'", '"')
    else:
        # Ganti kunci ber-single quote: {'name': ... -> {"name": ...
        s = re.sub(r"([{,]\s*)'([^']+)'\s*:", r'\1"\2":', s)
        # Ganti nilai ber-single quote: : 'value' -> : "value"
        s = re.sub(r":\s*'([^']*)'", r': "\1"', s)

    # 4. Tangani tanda petik ganda yang belum tertutup (abaikan escaped quote \")
    unescaped_quotes = len(re.findall(r'(?<!\\)"', s))
    if unescaped_quotes % 2 != 0:
        s += '"'

    # 5. Hapus trailing comma sekali lagi jika perbaikan tanda kutip menyebabkannya
    s = re.sub(r",\s*([\]}])", r"\1", s)
    s = re.sub(r",\s*$", "", s)

    # 6. Tangani kunci menggantung yang terpotong tanpa nilai (misal: "key":)
    if re.search(r':\s*$', s):
        s += " null"

    # 7. Seimbangkan tanda kurung penutup menggunakan STACK (LIFO) yang sadar string
    bracket_stack: List[str] = []
    in_str = False
    escape = False

    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str:
            if ch == '{':
                bracket_stack.append('}')
            elif ch == '[':
                bracket_stack.append(']')
            elif ch == '}':
                if bracket_stack and bracket_stack[-1] == '}':
                    bracket_stack.pop()
            elif ch == ']':
                if bracket_stack and bracket_stack[-1] == ']':
                    bracket_stack.pop()

    # Tutup kurung sesuai urutan LIFO stack
    while bracket_stack:
        s += bracket_stack.pop()

    # 8. PENTING: Bersihkan kembali koma yang berada di depan kurung penutup yang baru saja ditambahkan
    s = re.sub(r",\s*([\]}])", r"\1", s)

    return s


def parse_and_repair_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Mencoba mem-parsing JSON, dan jika gagal karena cacat sintaks dari LLM,
    melakukan perbaikan heuristik bertahap hingga menghasilkan dictionary valid.
    """
    cleaned = raw_text.strip()
    if not cleaned:
        return None

    # Percobaan 1: Parsing langsung
    try:
        res = json.loads(cleaned)
        if isinstance(res, dict):
            return res
        if isinstance(res, list) and res and isinstance(res[0], dict):
            return {"items": res}
    except json.JSONDecodeError:
        pass

    # Percobaan 2: Ekstraksi dari blok kode markdown ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        snippet = match.group(1).strip()
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            repaired = _repair_common_json_errors(snippet)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    # Percobaan 3: Perbaikan langsung jika teks diawali kurung kurawal/siku
    if cleaned.startswith(("{", "[")):
        repaired_full = _repair_common_json_errors(cleaned)
        try:
            parsed = json.loads(repaired_full)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return {"items": parsed}
        except json.JSONDecodeError:
            pass

    # Percobaan 4: Python ast.literal_eval fallback (untuk format pythonic dictionary)
    try:
        evaluated = ast.literal_eval(cleaned)
        if isinstance(evaluated, dict):
            return evaluated
    except (ValueError, SyntaxError):
        try:
            patched_eval = cleaned.rstrip().rstrip(",") + "}"
            evaluated = ast.literal_eval(patched_eval)
            if isinstance(evaluated, dict):
                return evaluated
        except (ValueError, SyntaxError):
            pass

    # Percobaan 5: Ekstraksi kurung kurawal terluar pertama { ... } (jika ada teks pengantar/penutup)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1:
        snippet = cleaned[start : end + 1] if (end != -1 and end > start) else cleaned[start:]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            repaired = _repair_common_json_errors(snippet)
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

    # Percobaan 6: Perbaikan seluruh teks secara menyeluruh (fallback akhir)
    repaired_full = _repair_common_json_errors(cleaned)
    try:
        parsed_full = json.loads(repaired_full)
        if isinstance(parsed_full, dict):
            return parsed_full
    except json.JSONDecodeError:
        pass

    return None


def repair_malformed_tool_call(raw_tool_calls: list) -> list:
    """
    Ketika LLM mengeluarkan argumen JSON yang terpotong, memiliki trailing comma,
    atau salah tanda kutip saat memanggil tool, metode ini melakukan parsing toleran
    dan menambal struktur JSON tersebut sebelum divalidasi oleh skema tool.

    Mendukung format:
    1. Standar OpenAI tool_calls:
       [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "..."}}]
    2. Format langsung:
       [{"name": "read_file", "arguments": "..."}, {"tool_id": "...", "parameters": "..."}]
    3. Format teks raw JSON atau blok markdown di dalam list.
    """
    if not raw_tool_calls:
        return []

    repaired_calls: List[Dict[str, Any]] = []

    for item in raw_tool_calls:
        if isinstance(item, dict):
            call_copy = dict(item)

            # Format 1: OpenAI tool_call {"id": ..., "function": {"name": ..., "arguments": ...}}
            if "function" in call_copy and isinstance(call_copy["function"], dict):
                fn = dict(call_copy["function"])
                raw_args = fn.get("arguments")

                if isinstance(raw_args, str):
                    repaired_dict = parse_and_repair_json(raw_args)
                    if repaired_dict is not None:
                        fn["arguments"] = json.dumps(repaired_dict)
                    else:
                        fn["arguments"] = _repair_common_json_errors(raw_args)
                elif isinstance(raw_args, dict):
                    # Sudah berupa dict, pastikan tetap valid
                    fn["arguments"] = json.dumps(raw_args)

                call_copy["function"] = fn
                repaired_calls.append(call_copy)

            # Format 2: Direct arguments {"name": ..., "arguments": ...}
            elif "arguments" in call_copy:
                raw_args = call_copy.get("arguments")
                if isinstance(raw_args, str):
                    repaired_dict = parse_and_repair_json(raw_args)
                    if repaired_dict is not None:
                        call_copy["arguments"] = json.dumps(repaired_dict)
                    else:
                        call_copy["arguments"] = _repair_common_json_errors(raw_args)
                repaired_calls.append(call_copy)

            # Format 3: Direct parameters {"tool_id": ..., "parameters": ...}
            elif "parameters" in call_copy:
                raw_params = call_copy.get("parameters")
                if isinstance(raw_params, str):
                    repaired_dict = parse_and_repair_json(raw_params)
                    if repaired_dict is not None:
                        call_copy["parameters"] = repaired_dict
                    else:
                        call_copy["parameters"] = _repair_common_json_errors(raw_params)
                repaired_calls.append(call_copy)

            else:
                repaired_calls.append(call_copy)

        elif isinstance(item, str):
            # Jika elemen berupa raw string (misal teks JSON pemanggilan tool)
            extracted = extract_tool_calls(item)
            if extracted:
                repaired_calls.extend(extracted)
            else:
                parsed = parse_and_repair_json(item)
                if parsed:
                    repaired_calls.append(parsed)
                else:
                    logger.warning(f"[ToolCallRepair] Gagal menambal tool call string: {item[:80]}...")
        else:
            repaired_calls.append(item)

    return repaired_calls


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    Mengekstraksi satu atau beberapa pemanggilan perkakas dari respons teks LLM.
    Mendukung format blok markdown ```json { ... } ``` maupun teks inline.
    """
    results: List[Dict[str, Any]] = []

    # 1. Cek format blok markdown JSON ```json ... ```
    matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    for m in matches:
        parsed = parse_and_repair_json(m)
        if isinstance(parsed, dict):
            tool_id = parsed.get("tool") or parsed.get("tool_id") or parsed.get("name")
            params = parsed.get("parameters") or parsed.get("arguments") or {}
            if tool_id:
                results.append({
                    "id": f"call_{len(results) + 1}",
                    "type": "function",
                    "function": {
                        "name": tool_id,
                        "arguments": json.dumps(params) if isinstance(params, dict) else str(params),
                    },
                })

    if results:
        return results

    # 2. Cek format inline JSON
    parsed = parse_and_repair_json(text)
    if isinstance(parsed, dict):
        tool_id = parsed.get("tool") or parsed.get("tool_id") or parsed.get("name")
        params = parsed.get("parameters") or parsed.get("arguments") or {}
        if tool_id:
            results.append({
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": tool_id,
                    "arguments": json.dumps(params) if isinstance(params, dict) else str(params),
                },
            })

    return results


def extract_reasoning_and_content(raw_text: str) -> Tuple[str, str]:
    """
    Memisahkan token pemikiran/reasoning (di dalam tag <think>...</think>) dari konten akhir.

    Mengembalikan:
        Tuple (reasoning_text, final_content).
    """
    think_match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        final_content = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        return reasoning, final_content
    return "", raw_text.strip()


__all__ = [
    "repair_malformed_tool_call",
    "parse_and_repair_json",
    "extract_tool_calls",
    "extract_reasoning_and_content",
]
