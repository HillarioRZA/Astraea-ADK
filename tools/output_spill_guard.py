"""
Universal 50KB Output Truncation & Disk Spill Guard for LegionADK.

Menjaga agar kembalian tool tidak melebihi 50 KB (51.200 bytes) saat masuk ke context window LLM.
Jika melampaui batas, memotong teks tepat pada batas aman UTF-8 (tanpa merusak multibyte characters),
menyimpan output mentah lengkap ke direktori .astraea_spill/, menyematkan penanda
[OUTPUT_TRUNCATED_EXCEEDED_50KB] beserta path berkas log, serta menjalankan pemangkas berkas
kadaluwarsa secara otomatis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
import uuid
from typing import Any, Optional, Union

# pyrefly: ignore[missing-import]
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger("LegionADK.tools.output_spill_guard")

# Batas keras 50 KB = 51.200 bytes
DEFAULT_MAX_OUTPUT_BYTES: int = 51_200
TRUNCATION_MARKER: str = "[OUTPUT_TRUNCATED_EXCEEDED_50KB]"
DEFAULT_SPILL_DIR_NAME: str = ".astraea_spill"


class SpillResult(BaseModel):
    """Hasil immutable dari pemrosesan output tool oleh Spill Guard."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str
    truncated: bool
    spill_path: Optional[str] = None
    total_bytes: int


def _get_default_spill_dir() -> str:
    """Mengembalikan path direktori spill default (.astraea_spill di root proyek)."""
    env_dir = os.environ.get("ASTRAEA_SPILL_DIR")
    if env_dir:
        return os.path.abspath(env_dir)
    return os.path.abspath(DEFAULT_SPILL_DIR_NAME)


def truncate_utf8_bytes(raw_bytes: bytes, max_bytes: int) -> bytes:
    """
    Memotong bytes agar panjangnya <= max_bytes tepat pada batas karakter UTF-8 yang valid.
    Tidak akan pernah menghasilkan urutan UTF-8 rusak atau memotong karakter multibyte di tengah.
    """
    if len(raw_bytes) <= max_bytes:
        return raw_bytes

    if max_bytes <= 0:
        return b""

    # Potong awal sebesar max_bytes
    sliced = raw_bytes[:max_bytes]

    # Mundur hingga menemukan batas decoding UTF-8 yang valid
    while sliced:
        try:
            sliced.decode("utf-8")
            return sliced
        except UnicodeDecodeError as exc:
            # Jika kegagalan decode terjadi pada akhir byte yang terpotong,
            # pangkas persis hingga batas karakter valid sebelum error
            if exc.start < len(sliced):
                sliced = sliced[:exc.start]
            else:
                sliced = sliced[:-1]

    return b""


def _write_file_sync(file_path: str, data: bytes) -> None:
    """Helper sinkron penulisan berkas biner."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(data)


def _read_file_sync(file_path: str) -> str:
    """Helper sinkron pembacaan berkas teks."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Spill log file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _sanitize_session_id(session_id: Optional[str]) -> str:
    """Membersihkan session_id atau handle agar aman digunakan sebagai nama file."""
    if not session_id:
        return uuid.uuid4().hex[:8]
    clean = re.sub(r"[^\w\-.]", "_", session_id.strip())
    clean = clean.strip("._")
    return clean[:32] if clean else uuid.uuid4().hex[:8]


def _prune_spill_cache_sync(
    spill_dir: str,
    max_files: int = 50,
    max_age_seconds: int = 86400,
) -> int:
    """
    Secara sinkron memangkas berkas spill yang berumur lebih dari max_age_seconds
    atau jika total jumlah berkas melebihi max_files (FIFO berdasarkan st_mtime).
    """
    if not os.path.exists(spill_dir):
        return 0

    now = time.time()
    pruned = 0
    entries = []

    try:
        filenames = os.listdir(spill_dir)
    except OSError as err:
        logger.warning(f"[SpillGuard] Gagal membaca direktori {spill_dir}: {err}")
        return 0

    for fname in filenames:
        if not fname.startswith("spill_") or not fname.endswith(".log"):
            continue
        fpath = os.path.join(spill_dir, fname)
        if os.path.isfile(fpath):
            try:
                st = os.stat(fpath)
                entries.append((st.st_mtime, fpath))
            except OSError:
                continue

    # 1. Hapus berkas yang telah melewati batas usia (TTL kadaluwarsa)
    remaining = []
    for mtime, fpath in entries:
        if (now - mtime) > max_age_seconds:
            try:
                os.remove(fpath)
                pruned += 1
            except OSError:
                pass
        else:
            remaining.append((mtime, fpath))

    # 2. Hapus berkas terlama jika jumlah tersisa masih melampaui kuota max_files
    if len(remaining) > max_files:
        remaining.sort(key=lambda x: x[0])
        excess = len(remaining) - max_files
        for i in range(excess):
            try:
                os.remove(remaining[i][1])
                pruned += 1
            except OSError:
                pass

    if pruned > 0:
        logger.debug(f"[SpillGuard] Berhasil memangkas {pruned} berkas spill kadaluwarsa di {spill_dir}.")
    return pruned


async def prune_spill_cache(
    spill_dir: Optional[str] = None,
    max_files: int = 50,
    max_age_seconds: int = 86400,
) -> int:
    """Memangkas berkas spill secara asinkron tanpa memblokir event loop."""
    target_dir = spill_dir or _get_default_spill_dir()
    return await asyncio.to_thread(
        _prune_spill_cache_sync,
        target_dir,
        max_files,
        max_age_seconds,
    )


def prune_spill_cache_sync(
    spill_dir: Optional[str] = None,
    max_files: int = 50,
    max_age_seconds: int = 86400,
) -> int:
    """Memangkas berkas spill kadaluwarsa secara sinkron."""
    target_dir = spill_dir or _get_default_spill_dir()
    return _prune_spill_cache_sync(target_dir, max_files, max_age_seconds)


def _normalize_to_bytes(raw_output: Any) -> bytes:
    """Mengonversi payload kembalian tool ke representasi bytes UTF-8."""
    if isinstance(raw_output, bytes):
        return raw_output
    if isinstance(raw_output, str):
        return raw_output.encode("utf-8")
    if isinstance(raw_output, (dict, list, tuple)):
        try:
            return json.dumps(raw_output, ensure_ascii=False, indent=2).encode("utf-8")
        except (TypeError, ValueError):
            return str(raw_output).encode("utf-8")
    return str(raw_output).encode("utf-8")


def process_output_sync(
    raw_output: Any,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    spill_dir: Optional[str] = None,
    session_id: Optional[str] = None,
) -> SpillResult:
    """
    Versi sinkron untuk mengevaluasi ukuran output tool:
    - Jika <= max_bytes: dikembalikan utuh tanpa truncating.
    - Jika > max_bytes: dipotong pada batas aman UTF-8, disimpan ke disk di .astraea_spill/,
      dan disematkan penanda [OUTPUT_TRUNCATED_EXCEEDED_50KB] beserta path berkas lengkap.
    """
    raw_bytes = _normalize_to_bytes(raw_output)
    total_bytes = len(raw_bytes)

    if total_bytes <= max_bytes:
        return SpillResult(
            content=raw_bytes.decode("utf-8", errors="replace"),
            truncated=False,
            spill_path=None,
            total_bytes=total_bytes,
        )

    # 1. Potong tepat pada batas aman UTF-8
    safe_truncated_bytes = truncate_utf8_bytes(raw_bytes, max_bytes)
    truncated_text = safe_truncated_bytes.decode("utf-8", errors="ignore")

    # 2. Siapkan direktori dan berkas spill log
    base_dir = spill_dir or _get_default_spill_dir()
    sess_tag = _sanitize_session_id(session_id)
    timestamp = int(time.time())
    rand_suffix = uuid.uuid4().hex[:6]
    filename = f"spill_{sess_tag}_{timestamp}_{rand_suffix}.log"
    spill_path = os.path.abspath(os.path.join(base_dir, filename))

    # 3. Tulis seluruh raw_bytes mentah ke disk
    _write_file_sync(spill_path, raw_bytes)
    logger.warning(
        f"[SpillGuard] Output tool melampaui batas aman ({total_bytes} bytes > {max_bytes} bytes). "
        f"Output lengkap disimpan ke: {spill_path}"
    )

    # 4. Sematkan penanda beserta path berkas log lengkap
    content_with_marker = (
        f"{truncated_text}\n\n"
        f"{TRUNCATION_MARKER}\n"
        f"Full output ({total_bytes} bytes) spilled to: {spill_path}\n"
    )

    # 5. Otomatis pangkas berkas kadaluwarsa
    _prune_spill_cache_sync(base_dir)

    return SpillResult(
        content=content_with_marker,
        truncated=True,
        spill_path=spill_path,
        total_bytes=total_bytes,
    )


async def process_output(
    raw_output: Any,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    spill_dir: Optional[str] = None,
    session_id: Optional[str] = None,
) -> SpillResult:
    """
    Versi asinkron untuk mengevaluasi ukuran output tool:
    Menjalankan operasi I/O disk di threadpool terpisah via asyncio.to_thread.
    """
    raw_bytes = _normalize_to_bytes(raw_output)
    total_bytes = len(raw_bytes)

    if total_bytes <= max_bytes:
        return SpillResult(
            content=raw_bytes.decode("utf-8", errors="replace"),
            truncated=False,
            spill_path=None,
            total_bytes=total_bytes,
        )

    # 1. Potong tepat pada batas aman UTF-8
    safe_truncated_bytes = truncate_utf8_bytes(raw_bytes, max_bytes)
    truncated_text = safe_truncated_bytes.decode("utf-8", errors="ignore")

    # 2. Siapkan direktori dan berkas spill log
    base_dir = spill_dir or _get_default_spill_dir()
    sess_tag = _sanitize_session_id(session_id)
    timestamp = int(time.time())
    rand_suffix = uuid.uuid4().hex[:6]
    filename = f"spill_{sess_tag}_{timestamp}_{rand_suffix}.log"
    spill_path = os.path.abspath(os.path.join(base_dir, filename))

    # 3. Tulis seluruh raw_bytes ke disk via threadpool
    await asyncio.to_thread(_write_file_sync, spill_path, raw_bytes)
    logger.warning(
        f"[SpillGuard] Output tool melampaui batas aman ({total_bytes} bytes > {max_bytes} bytes). "
        f"Output lengkap disimpan ke: {spill_path}"
    )

    # 4. Sematkan penanda beserta path berkas log lengkap
    content_with_marker = (
        f"{truncated_text}\n\n"
        f"{TRUNCATION_MARKER}\n"
        f"Full output ({total_bytes} bytes) spilled to: {spill_path}\n"
    )

    # 5. Otomatis pangkas berkas kadaluwarsa
    await prune_spill_cache(base_dir)

    return SpillResult(
        content=content_with_marker,
        truncated=True,
        spill_path=spill_path,
        total_bytes=total_bytes,
    )


async def guard_tool_output(
    raw_output: Any,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    spill_dir: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    Universal Asynchronous Guard:
    Mengembalikan teks aman yang dijamin <= 50 KB (+ marker penanda jika terpotong).
    """
    res = await process_output(
        raw_output=raw_output,
        max_bytes=max_bytes,
        spill_dir=spill_dir,
        session_id=session_id,
    )
    return res.content


def guard_tool_output_sync(
    raw_output: Any,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    spill_dir: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    Universal Synchronous Guard:
    Mengembalikan teks aman yang dijamin <= 50 KB (+ marker penanda jika terpotong).
    """
    res = process_output_sync(
        raw_output=raw_output,
        max_bytes=max_bytes,
        spill_dir=spill_dir,
        session_id=session_id,
    )
    return res.content


async def read_spilled_output(spill_path: str) -> str:
    """Membaca kembali isi lengkap berkas spill dari disk secara asinkron."""
    return await asyncio.to_thread(_read_file_sync, spill_path)


def read_spilled_output_sync(spill_path: str) -> str:
    """Membaca kembali isi lengkap berkas spill dari disk secara sinkron."""
    return _read_file_sync(spill_path)


__all__ = [
    "DEFAULT_MAX_OUTPUT_BYTES",
    "TRUNCATION_MARKER",
    "DEFAULT_SPILL_DIR_NAME",
    "SpillResult",
    "truncate_utf8_bytes",
    "process_output",
    "process_output_sync",
    "guard_tool_output",
    "guard_tool_output_sync",
    "read_spilled_output",
    "read_spilled_output_sync",
    "prune_spill_cache",
    "prune_spill_cache_sync",
]
