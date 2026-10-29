"""
Model Router untuk LegionADK.
Menyediakan orkestrasi multi-provider, in-memory RPM/TPM throttling, auto-cooldown (429/5xx),
kalkulasi statistik penggunaan token murni, dan Failover Cascade otonom tanpa respons palsu/mock.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

# pyrefly: ignore[missing-import]
import httpx

from LegionADK.gateway.adapter.anthropic_adapter import AnthropicAdapter
from LegionADK.gateway.adapter.base_adapter import BaseLLMAdapter
from LegionADK.gateway.adapter.custom_adapter import CustomAdapter
from LegionADK.gateway.adapter.gemini_adapter import GeminiAdapter
from LegionADK.gateway.adapter.openai_compatible import OpenAICompatibleAdapter
from LegionADK.gateway.cache_boundary import sort_tools_alphabetically

if TYPE_CHECKING:
    from LegionADK.core.llm_spec import CancellationToken

logger = logging.getLogger("LegionADK.gateway.model_router")


class UpstreamChannel:
    """Manajer kanal provider hulu dengan kontrol kuota RPM/TPM di memori dan auto-cooldown."""

    def __init__(
        self,
        provider: str,
        model: str,
        rpm_limit: int = 30,
        tpm_limit: int = 60000,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.cooldown_until: Optional[float] = None
        self.request_timestamps: List[float] = []
        self.token_window: List[List[Any]] = []
        self._lock = asyncio.Lock()

    def is_healthy(self) -> bool:
        """Memeriksa apakah kanal sedang bebas dari masa cooldown."""
        if self.cooldown_until:
            if time.time() < self.cooldown_until:
                return False
            self.cooldown_until = None
        return True

    def trigger_cooldown(self, seconds: int = 60) -> None:
        """Membekukan kanal selama sejumlah detik akibat 429 atau 5xx."""
        self.cooldown_until = time.time() + seconds
        logger.warning(
            f"[ModelRouter] Kanal [{self.provider}/{self.model}] dibekukan (cooldown) "
            f"selama {seconds} detik."
        )

    def record_actual_tokens(
        self,
        ticket: Any,
        actual_tokens: int,
        estimated_tokens: Optional[int] = None,
    ) -> None:
        """
        Mengoreksi estimasi token pada tiket request tertentu dengan nilai token aktual dari vendor.
        """
        if isinstance(ticket, list) and len(ticket) >= 2:
            ticket[1] = actual_tokens
        elif self.token_window and estimated_tokens is not None:
            diff = actual_tokens - estimated_tokens
            last_time, last_tok = self.token_window[-1]
            self.token_window[-1] = [last_time, max(0, last_tok + diff)]

    async def throttle_if_needed(
        self,
        estimated_tokens: int = 2000,
        token: Optional[CancellationToken] = None,
    ) -> List[Any]:
        """
        Memeriksa kuota RPM/TPM di memori secara coroutine-safe.
        Lock dilepas saat sleep untuk mencegah head-of-line blocking pada coroutine lain.
        Mengembalikan tiket [timestamp, tokens] yang dapat direkonsiliasi secara presisi.
        """
        while True:
            if token is not None:
                await token.wait_if_paused()
                token.raise_if_cancelled()

            wait_time = 0.0
            async with self._lock:
                now = time.time()
                # Pangkas jejak request & token yang lebih lama dari 60 detik
                self.request_timestamps = [t for t in self.request_timestamps if now - t < 60.0]
                self.token_window = [e for e in self.token_window if now - e[0] < 60.0]

                current_rpm = len(self.request_timestamps)
                current_tpm = sum(e[1] for e in self.token_window)

                rpm_exceeded = current_rpm >= self.rpm_limit
                tpm_exceeded = (current_tpm + estimated_tokens) > self.tpm_limit

                if not rpm_exceeded and not tpm_exceeded:
                    rec_now = time.time()
                    ticket = [rec_now, estimated_tokens]
                    self.request_timestamps.append(rec_now)
                    self.token_window.append(ticket)
                    return ticket

                # Hitung estimasi waktu tunggu optimal berdasarkan rekaman tertua dalam window
                if self.request_timestamps or self.token_window:
                    oldest_req = self.request_timestamps[0] if self.request_timestamps else now
                    oldest_tok = self.token_window[0][0] if self.token_window else now
                    oldest = min(oldest_req, oldest_tok)
                    wait_time = max(0.1, min(5.0, 60.0 - (now - oldest) + 0.1))
                else:
                    wait_time = 0.5

                logger.info(
                    f"[ModelRouter] Throttling aktif pada [{self.provider}/{self.model}]: "
                    f"RPM {current_rpm}/{self.rpm_limit}, TPM {current_tpm}/{self.tpm_limit}. "
                    f"Menunda {wait_time:.2f}s..."
                )

            # Sleep dilakukan di luar lock untuk menghindari head-of-line blocking
            await asyncio.sleep(wait_time)


class ModelRouter:
    """
    Router terpadu tingkat enterprise untuk routing LLM multi-kanal.
    Mengatur pemilihan provider utama, failover cascade otomatis saat 429/5xx,
    dan perhitungan biaya token tanpa pernah mengembalikan respon simulasi/palsu.
    """

    def __init__(self) -> None:
        self.openai_compat = OpenAICompatibleAdapter()
        self.anthropic = AnthropicAdapter()
        self.gemini = GeminiAdapter()
        self.custom = CustomAdapter()
        self.channels: Dict[str, UpstreamChannel] = {}
        self.provider_cooldowns: Dict[str, float] = {}

    def _get_channel(self, provider: str, model: str) -> UpstreamChannel:
        """Mengambil atau menginisialisasi UpstreamChannel untuk pasangan provider dan model."""
        key = f"{provider.lower()}:{model}"
        if key not in self.channels:
            m_lower = model.lower()
            rpm = 15 if "free" in m_lower else 45
            tpm = 30000 if "free" in m_lower else 100000
            ch = UpstreamChannel(provider, model, rpm_limit=rpm, tpm_limit=tpm)
            # Terapkan cooldown tingkat provider jika sedang aktif
            p_lower = provider.lower()
            if p_lower in self.provider_cooldowns:
                cooldown_expiry = self.provider_cooldowns[p_lower]
                if time.time() < cooldown_expiry:
                    ch.cooldown_until = cooldown_expiry
            self.channels[key] = ch
        return self.channels[key]

    def trigger_cooldown(self, provider_id: str, seconds: int = 60) -> None:
        """
        Membekukan provider tertentu selama N detik.
        provider_id dapat berformat 'provider:model' atau 'provider'.
        """
        now = time.time()
        if ":" in provider_id:
            prov, mdl = provider_id.split(":", 1)
            channel = self._get_channel(prov, mdl)
            channel.trigger_cooldown(seconds=seconds)
        else:
            p_lower = provider_id.lower()
            self.provider_cooldowns[p_lower] = now + seconds
            # Bekukan seluruh kanal yang sudah terdaftar menggunakan provider tersebut
            for ch in self.channels.values():
                if ch.provider == p_lower:
                    ch.trigger_cooldown(seconds=seconds)

    def _resolve_adapter(self, provider: str) -> BaseLLMAdapter:
        """
        Menentukan adapter berdasarkan provider resmi (gemini, openai, anthropic, ollama, vllm).
        Selain itu dikategorikan sebagai custom.
        """
        p = provider.lower()
        if p == "gemini":
            return self.gemini
        elif p == "openai":
            return self.openai_compat
        elif p == "anthropic":
            return self.anthropic
        elif p in ("ollama", "vllm"):
            return self.openai_compat
        else:
            return self.custom

    def _resolve_credentials(self, provider: str) -> Tuple[str, str]:
        """
        Membaca kredensial API key dan Base URL dari environment variable.
        Mendukung provider resmi (gemini, openai, anthropic, ollama, vllm).
        Selain itu dikategorikan sebagai custom (CUSTOM_PROVIDER_API_KEY & CUSTOM_PROVIDER_BASE_URL).
        """
        p = provider.lower()
        if p == "gemini":
            return (
                os.getenv("GEMINI_API_KEY", ""),
                os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
            )
        elif p == "openai":
            return (
                os.getenv("OPENAI_API_KEY", ""),
                os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )
        elif p == "anthropic":
            return (
                os.getenv("ANTHROPIC_API_KEY", ""),
                os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
            )
        elif p == "ollama":
            return (
                os.getenv("OLLAMA_API_KEY", ""),
                os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            )
        elif p == "vllm":
            return (
                os.getenv("VLLM_API_KEY", ""),
                os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            )
        else:
            return (
                os.getenv("CUSTOM_PROVIDER_API_KEY", ""),
                os.getenv("CUSTOM_PROVIDER_BASE_URL", ""),
            )

    @staticmethod
    def calculate_token_usage(
        prompt_tokens: int,
        cached_prompt_tokens: int,
        completion_tokens: int,
    ) -> Dict[str, int]:
        """
        Menghitung rincian total penggunaan token murni (tanpa komponen harga).
        Mengembalikan kamus berisi prompt_tokens, cached_prompt_tokens,
        completion_tokens, dan total_tokens.
        """
        total_tokens = prompt_tokens + completion_tokens
        return {
            "prompt_tokens": prompt_tokens,
            "cached_prompt_tokens": cached_prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    async def execute_routed_completion(
        self,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: str = "",
        primary_provider: str = "custom",
        primary_model: str = "deepseek-v4.1-flash",
        fallback_models: Optional[List[Union[str, Tuple[str, str]]]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = 4096,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        images: Optional[List[str]] = None,
        thinking_level: str = "medium",
        token: Optional[CancellationToken] = None,
    ) -> Dict[str, Any]:
        """
        Menjalankan panggilan completion dengan:
        1. Pemeriksaan kuota RPM/TPM di memori.
        2. Auto-cooldown saat terjadi HTTP 429 atau 5xx via trigger_cooldown.
        3. Failover Cascade ke model cadangan tanpa memutus alur kerja agen.
        4. Melempar error HTTP asli jika seluruh kandidat gagal (TIDAK ADA respon palsu).
        """
        candidates: List[Tuple[str, str]] = [(primary_provider, primary_model)]

        if fallback_models:
            for fb in fallback_models:
                if isinstance(fb, tuple):
                    candidates.append((fb[0], fb[1]))
                elif isinstance(fb, str):
                    if "/" in fb:
                        prov, mdl = fb.split("/", 1)
                    else:
                        prov, mdl = "custom", fb
                    candidates.append((prov, mdl))

        last_error: Optional[Exception] = None

        for prov, mdl in candidates:
            # Hormati CancellationToken sebelum berpindah kandidat
            if token is not None:
                await token.wait_if_paused()
                token.raise_if_cancelled()

            channel = self._get_channel(prov, mdl)

            # 1. Periksa apakah provider sedang dalam masa cooldown
            if not channel.is_healthy():
                logger.warning(
                    f"[ModelRouter] Kanal [{prov}/{mdl}] sedang cooldown. "
                    f"Melakukan failover ke kandidat berikutnya..."
                )
                continue

            # 2. Periksa kuota RPM/TPM Throttling di memori
            if prompt:
                text_len = len(prompt)
            elif messages:
                text_len = sum(len(str(m.get("content", ""))) for m in messages)
            else:
                text_len = 6000
            if system_prompt:
                text_len += len(system_prompt)

            estimated_tokens = max(1, text_len // 4)
            ticket = await channel.throttle_if_needed(estimated_tokens=estimated_tokens, token=token)

            adapter = self._resolve_adapter(prov)
            api_key, base_url = self._resolve_credentials(prov)

            # Pastikan urutan tools byte-stable sebelum diteruskan ke vendor API
            sorted_tools = sort_tools_alphabetically(tools)[0] if tools else None

            try:
                logger.info(f"[ModelRouter] Mengirim panggilan ke [{prov}] model [{mdl}]...")
                result = await adapter.complete(
                    prompt=prompt,
                    messages=messages,
                    system_prompt=system_prompt,
                    model=mdl,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=sorted_tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    images=images,
                    thinking_level=thinking_level,
                    token=token,
                )

                # Rekonsiliasi estimasi token tiket ini dengan token aktual vendor
                actual_prompt = result.get("prompt_tokens", 0)
                actual_completion = result.get("completion_tokens", 0)
                actual_total = actual_prompt + actual_completion
                if actual_total > 0:
                    channel.record_actual_tokens(ticket=ticket, actual_tokens=actual_total, estimated_tokens=estimated_tokens)

                # Kalkulasi statistik penggunaan token
                token_usage = self.calculate_token_usage(
                    prompt_tokens=actual_prompt,
                    cached_prompt_tokens=result.get("cached_prompt_tokens", 0),
                    completion_tokens=actual_completion,
                )
                result["token_usage"] = token_usage
                return result

            except httpx.HTTPStatusError as http_err:
                status_code = http_err.response.status_code
                logger.warning(
                    f"[ModelRouter] Kanal [{prov}/{mdl}] mengembalikan HTTP {status_code}: "
                    f"{http_err.response.text}"
                )

                # Jika error 429 (Rate Limit) atau 5xx (Server Error), otomatis cooldown & failover
                if status_code in (429, 500, 502, 503, 504):
                    self.trigger_cooldown(f"{prov}:{mdl}", seconds=60)

                last_error = http_err

            except (httpx.TimeoutException, httpx.RequestError) as net_err:
                logger.warning(
                    f"[ModelRouter] Kanal [{prov}/{mdl}] mengalami gangguan jaringan/timeout: {net_err}"
                )
                self.trigger_cooldown(f"{prov}:{mdl}", seconds=30)
                last_error = net_err

            except asyncio.CancelledError:
                # Pembatalan pengguna harus langsung dihentikan tanpa failover
                raise

            except Exception as err:
                logger.warning(f"[ModelRouter] Kanal [{prov}/{mdl}] mengalami error tak terduga: {err}")
                last_error = err

        # Jika seluruh kandidat gagal, lempar error asli
        if last_error is not None:
            raise last_error

        raise RuntimeError(
            f"Semua kanal model ({len(candidates)} kandidat) gagal dieksekusi tanpa respon yang valid."
        )


__all__ = [
    "UpstreamChannel",
    "ModelRouter",
]
