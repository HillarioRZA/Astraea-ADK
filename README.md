# Astraea ADK (Agent Development Kit)

Astraea ADK adalah toolkit tingkat enterprise untuk perancangan, orkestrasi, dan komunikasi sistem multi-agen berbasis **Mixture-of-Agents (MoA)** dengan fokus pada determinisme, performa tinggi, dan efisiensi biaya inferensi LLM.

---

## 🌟 Fitur Utama

- **Byte-Stable Cache Boundary (`cache_boundary.py`)**:
  - Penjaga prompt caching byte-stable 100% in-memory (RAM) menggunakan hashing SHA-256.
  - Memisahkan *static prefix* (soul prompt, instruksi peran, skema tool alfabetis) dari *dynamic conversation context* via `<!-- LEGION_CACHE_BOUNDARY -->`.
  - Memaksimalkan **Cache HIT** pada penyedia LLM (Anthropic, Gemini, OpenAI, DeepSeek) untuk memangkas latensi hingga 90% dan menghemat biaya input token.

- **Fault-Tolerant Tool Call Repair (`tool_call_repair.py`)**:
  - Mesin penambal sintaks JSON toleran untuk menangani argumen panggilan fungsi yang rusak atau terpotong akibat `max_tokens` atau pemutusan koneksi streaming.
  - Memperbaiki kutip tunggal Python/JS, literal boolean (`True`/`False`/`None`), koma menggantung (*trailing commas*), serta menyeimbangkan kurung kurawal/siku yang belum tertutup.

- **Unified Model Router & Resilient Failover (`model_router.py`)**:
  - Orkestrasi multi-kanal dengan proteksi kuota RPM/TPM in-memory.
  - **Auto-Cooldown otonom**: Pembekuan kanal otomatis selama $N$ detik jika terkena HTTP 429 atau 5xx, dilanjutkan dengan *Failover Cascade* ke model cadangan tanpa memutus alur kerja agen.
  - **Dukungan Provider Resmi**: Gemini, OpenAI, Anthropic, Ollama, vLLM.
  - **Custom Provider Universal**: Mendukung endpoint privat, proxy lokal, atau agregator pihak ketiga via `CUSTOM_PROVIDER_BASE_URL` dan `CUSTOM_PROVIDER_API_KEY`.
  - Menghitung statistik penggunaan token murni tanpa melempar respon tiruan/mock palsu.

- **Immutable Core Spec & Concurrency Tokens (`llm_spec.py`)**:
  - Definisi peran agen deterministik (`LLMSpec`) dengan validasi ketat Pydantic v2.
  - Amplop pesan antar-agen (`MessageEnvelope`) yang dapat di-replay dan diaudit.
  - Pengendali konkuren asinkron (`CancellationToken`) untuk operasi jeda (*pause/resume*) dan pembatalan non-blocking.

---

## 📁 Struktur Repositori

```text
Astraea-ADK/
├── core/
│   ├── __init__.py
│   └── llm_spec.py             # LLMSpec, MessageEnvelope, MoAMetadata, CancellationToken
├── gateway/
│   ├── __init__.py
│   ├── cache_boundary.py       # Byte-stable cache boundary & in-memory SHA-256
│   ├── model_router.py         # Multi-provider routing, in-memory rate-limiter, failover
│   ├── tool_call_repair.py     # Fault-tolerant JSON repair engine
│   └── adapter/
│       ├── __init__.py
│       ├── base_adapter.py     # BaseLLMAdapter ABC interface
│       ├── openai_compatible.py# OpenAI-compatible REST adapter
│       ├── anthropic_adapter.py# Anthropic Messages REST adapter
│       ├── gemini_adapter.py   # Google Gemini REST adapter
│       └── custom_adapter.py   # Universal custom provider adapter
├── memory/                     # Memory & state management layer
├── protocol/                   # Protocol & inter-agent communication
├── runtime/                    # Agent execution runtime
└── tools/                      # Tool registry and execution
```

---

## 🚀 Konfigurasi Lingkungan (.env)

Untuk menggunakan penyedia kustom atau provider resmi, siapkan berkas `.env`:

```env
# Custom Provider (Aggregator / Private Proxy / Self-hosted)
CUSTOM_PROVIDER_API_KEY=your_api_key_here
CUSTOM_PROVIDER_BASE_URL=https://your-custom-endpoint.com/v1

# Official Providers (Opsional)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIzaSy...
OLLAMA_BASE_URL=http://localhost:11434/v1
VLLM_BASE_URL=http://localhost:8000/v1
```

---

## 📄 Lisensi
 
Dilindungi di bawah lisensi Proprietary & Confidential. Hak Cipta (c) 2025 Astraea ADK Contributors. Seluruh hak cipta dilindungi undang-undang.

