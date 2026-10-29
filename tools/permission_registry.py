"""
In-Memory Tool Registry & RBAC Guard for LegionADK.

Menyediakan:
1. ToolSpec: Kontrak Pydantic v2 deep-frozen yang menyimpan name (wajib berformat namespace:action),
   description, parameters_schema, risk_tier (0, 1, 2), dan handler (sync maupun async).
2. In-memory Registry: Pendaftaran dan manajemen tools murni di RAM tanpa dependensi database.
3. authorize_tool_call: Verifikasi hak akses terhadap spec.tool_permissions (exact, namespace wildcard, universal wildcard).
4. export_permitted_tools: Ekspor deklarasi fungsi OpenAI/Anthropic terurut alfabetis deterministik.
5. execute_registered_tool: Pipeline eksekusi aman (RBAC -> CancellationToken -> Argument Validation -> Handler Execution -> Output Spill Guard).
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import re
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
    overload,
)

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from LegionADK.core.llm_spec import CancellationToken, LLMSpec
from LegionADK.tools.output_spill_guard import guard_tool_output

logger = logging.getLogger("LegionADK.tools.permission_registry")


# ============================================================================
# 1. Custom Exceptions
# ============================================================================

class ToolRegistryError(Exception):
    """Base exception untuk seluruh operasi pada Tool Registry & RBAC Guard."""
    pass


class ToolPermissionError(ToolRegistryError, PermissionError):
    """Memicu error saat agen tidak memiliki izin RBAC untuk memanggil tool."""
    pass


class ToolNotFoundError(ToolRegistryError, KeyError):
    """Memicu error saat tool yang dipanggil tidak terdaftar di in-memory registry."""
    pass


class ToolArgumentError(ToolRegistryError, ValueError):
    """Memicu error saat parameter/argumen pemanggilan tool tidak valid."""
    pass


class ToolExecutionError(ToolRegistryError, RuntimeError):
    """Memicu error saat handler eksekusi fungsi tool gagal / melempar exception."""
    pass


# ============================================================================
# 2. Deep-Freeze Utilities
# ============================================================================

def _deep_freeze(obj: Any) -> Any:
    """Membekukan dictionary dan list secara rekursif menjadi MappingProxyType dan tuple."""
    if isinstance(obj, (dict, MappingProxyType)):
        return MappingProxyType({k: _deep_freeze(v) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return tuple(_deep_freeze(x) for x in obj)
    return obj


def _deep_unfreeze(obj: Any) -> Any:
    """Mengonversi MappingProxyType dan tuple kembali ke dict dan list murni."""
    if isinstance(obj, (MappingProxyType, dict)):
        return {k: _deep_unfreeze(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_deep_unfreeze(x) for x in obj]
    return obj


# Regex untuk memvalidasi format name (namespace:action)
TOOL_NAME_REGEX = re.compile(r"^[a-zA-Z0-9_-]+:[a-zA-Z0-9_.-]+$")


# ============================================================================
# 3. ToolSpec Contract (Pydantic v2 Deep-Frozen)
# ============================================================================

class ToolSpec(BaseModel):
    """
    Kontrak spesifikasi perkakas (Tool) LegionADK.
    Deep-frozen dan immutable untuk menjamin konsistensi state dan thread-safety.
    """
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    name: str = Field(
        ...,
        description="Nama tool berformat 'namespace:action', contoh: 'graph:locate', 'comm:mention'"
    )
    description: str = Field(
        ...,
        description="Penjelasan fungsionalitas dan instruksi pemakaian tool"
    )
    parameters_schema: Mapping[str, Any] = Field(
        default_factory=lambda: MappingProxyType({"type": "object", "properties": {}, "required": []}),
        description="JSON Schema valid untuk parameter tool (read-only mapping)"
    )
    risk_tier: Literal[0, 1, 2] = Field(
        default=0,
        description="Tingkat risiko: 0 (read-only/aman), 1 (moderat/write), 2 (kritis/eksekusi kode/eval)"
    )
    handler: Callable[..., Any] = Field(
        ...,
        description="Fungsi Python pelaksana (sync maupun async callable)"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        cleaned = v.strip()
        if not TOOL_NAME_REGEX.match(cleaned):
            raise ValueError(
                f"Nama tool '{v}' tidak valid. Wajib berformat 'namespace:action' "
                f"(contoh: 'graph:locate', 'comm:mention', 'builtin:read_file')."
            )
        return cleaned

    @field_validator("parameters_schema", mode="before")
    @classmethod
    def freeze_parameters_schema(cls, v: Any) -> Mapping[str, Any]:
        if v is None:
            v = {"type": "object", "properties": {}, "required": []}
        return _deep_freeze(v)

    @field_serializer("parameters_schema", when_used="always")
    def serialize_parameters_schema(self, v: Any) -> Dict[str, Any]:
        return _deep_unfreeze(v)


# ============================================================================
# 4. In-Memory Registry Storage & Registration Helpers
# ============================================================================

_TOOL_REGISTRY: Dict[str, ToolSpec] = {}


def get_registered_tool(tool_name: str) -> Optional[ToolSpec]:
    """Mengambil ToolSpec dari registry berdasarkan nama tool."""
    return _TOOL_REGISTRY.get(tool_name.strip())


def list_registered_tools() -> List[ToolSpec]:
    """Mengembalikan daftar seluruh ToolSpec yang terdaftar di in-memory registry."""
    return list(_TOOL_REGISTRY.values())


def unregister_tool(tool_name: str) -> bool:
    """Menghapus tool dari registry berdasarkan nama. Mengembalikan True jika ada yang dihapus."""
    cleaned = tool_name.strip()
    if cleaned in _TOOL_REGISTRY:
        del _TOOL_REGISTRY[cleaned]
        logger.info(f"[PermissionRegistry] Tool '{cleaned}' berhasil dihapus dari registry.")
        return True
    return False


def clear_tool_registry() -> None:
    """Mengosongkan seluruh isi registry in-memory (berguna untuk isolasi testing)."""
    _TOOL_REGISTRY.clear()
    logger.debug("[PermissionRegistry] Tool registry telah dikosongkan.")


@overload
def register_tool(tool: ToolSpec) -> ToolSpec: ...

@overload
def register_tool(
    tool: None = None,
    *,
    name: str,
    description: str,
    parameters_schema: Optional[Dict[str, Any]] = None,
    risk_tier: Literal[0, 1, 2] = 0,
    handler: Optional[Callable[..., Any]] = None,
) -> Union[ToolSpec, Callable[[Callable[..., Any]], ToolSpec]]: ...


def register_tool(
    tool: Optional[Union[ToolSpec, Callable[..., Any]]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters_schema: Optional[Dict[str, Any]] = None,
    risk_tier: Literal[0, 1, 2] = 0,
    handler: Optional[Callable[..., Any]] = None,
) -> Union[ToolSpec, Callable[[Callable[..., Any]], ToolSpec]]:
    """
    Mendaftarkan tool ke in-memory registry. Mendukung 3 mode pemanggilan:
    1. register_tool(spec: ToolSpec) -> ToolSpec
    2. register_tool(name="...", description="...", handler=my_fn, ...) -> ToolSpec
    3. @register_tool(name="...", description="...", ...) def my_fn(...) -> ToolSpec (decorator)
    """
    # Kasus 1: Diberikan langsung objek ToolSpec
    if isinstance(tool, ToolSpec):
        _TOOL_REGISTRY[tool.name] = tool
        logger.info(f"[PermissionRegistry] Tool '{tool.name}' (tier {tool.risk_tier}) didaftarkan.")
        return tool

    # Kasus 2: Diberikan handler langsung via keyword argument atau argumen pertama
    actual_handler = handler or (tool if callable(tool) else None)

    def decorator(fn: Callable[..., Any]) -> ToolSpec:
        if not name:
            raise ValueError("Parameter 'name' wajib disertakan saat mendaftarkan tool.")
        if not description:
            # Gunakan docstring fungsi jika ada
            fn_desc = inspect.getdoc(fn)
            if not fn_desc:
                raise ValueError(f"Deskripsi untuk tool '{name}' tidak boleh kosong.")
            desc = fn_desc.strip()
        else:
            desc = description.strip()

        schema = parameters_schema or {"type": "object", "properties": {}, "required": []}
        spec = ToolSpec(
            name=name,
            description=desc,
            parameters_schema=schema,
            risk_tier=risk_tier,
            handler=fn,
        )
        _TOOL_REGISTRY[spec.name] = spec
        logger.info(f"[PermissionRegistry] Tool '{spec.name}' (tier {spec.risk_tier}) didaftarkan via decorator.")
        return spec

    if actual_handler is not None and callable(actual_handler):
        return decorator(actual_handler)

    # Kasus 3: Dipanggil sebagai decorator dengan argumen: @register_tool(name="...", ...)
    return decorator


# ============================================================================
# 5. RBAC Authorization Engine
# ============================================================================

def authorize_tool_call(spec: LLMSpec, tool_name: str) -> bool:
    """
    Mengecek izin pemanggilan terhadap spec.tool_permissions di memori.
    Mendukung:
    - Pencocokan persis ("graph:locate" == "graph:locate")
    - Pencocokan wildcard namespace ("graph:*" cocok dengan "graph:locate", "graph:query")
    - Pencocokan universal wildcard ("*" atau "all" mengizinkan semua tool)

    Jika spec.tool_permissions bernilai None atau kosong, mengembalikan False.
    """
    if not spec.tool_permissions:
        return False

    clean_tool = tool_name.strip()
    tool_ns = clean_tool.split(":", 1)[0].lower() if ":" in clean_tool else ""

    for perm in spec.tool_permissions:
        p = perm.strip()
        # 1. Universal wildcard
        if p in ("*", "all"):
            return True

        # 2. Exact match (case-insensitive)
        if p.lower() == clean_tool.lower():
            return True

        # 3. Namespace wildcard (e.g. "graph:*")
        if p.endswith(":*"):
            perm_ns = p[:-2].strip().lower()
            if perm_ns and perm_ns == tool_ns:
                return True

    return False


# ============================================================================
# 6. Permitted Tools Export (Deterministic Schema for Prompt Cache)
# ============================================================================

def export_permitted_tools(
    spec: LLMSpec,
    format: Literal["openai", "anthropic"] = "openai",
) -> List[Dict[str, Any]]:
    """
    Mengekspor skema function declaration hanya untuk tools yang diizinkan bagi spec tersebut.
    Hasil diurutkan secara alfabetis deterministik berdasarkan nama tool agar menyatu
    dan kompatibel 100% dengan cache_boundary.py (byte-stable prompt caching).
    """
    permitted_tools: List[ToolSpec] = [
        tool for tool in _TOOL_REGISTRY.values()
        if authorize_tool_call(spec, tool.name)
    ]

    # Urutkan alfabetis deterministik berdasarkan nama tool
    sorted_tools = sorted(permitted_tools, key=lambda t: t.name.lower())

    exported: List[Dict[str, Any]] = []
    for t in sorted_tools:
        raw_schema = _deep_unfreeze(t.parameters_schema)
        if format == "anthropic":
            exported.append({
                "name": t.name,
                "description": t.description,
                "input_schema": raw_schema,
            })
        else:
            # Default OpenAI Compatible format
            exported.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": raw_schema,
                },
            })

    return exported


# ============================================================================
# 7. Safe Tool Execution Pipeline
# ============================================================================

def _validate_tool_arguments(tool: ToolSpec, arguments: Dict[str, Any]) -> None:
    """Memvalidasi parameter wajib (required) yang terdefinisi pada schema."""
    schema = _deep_unfreeze(tool.parameters_schema)
    required_fields = schema.get("required", [])
    if isinstance(required_fields, (list, tuple)):
        for req in required_fields:
            if req not in arguments:
                raise ToolArgumentError(
                    f"Argumen wajib '{req}' tidak ditemukan pada pemanggilan tool '{tool.name}'."
                )


def _format_raw_output(raw_result: Any) -> str:
    """Mengonversi hasil keluaran fungsi handler ke representasi string."""
    if isinstance(raw_result, str):
        return raw_result
    if isinstance(raw_result, bytes):
        return raw_result.decode("utf-8", errors="replace")
    if isinstance(raw_result, (dict, list, tuple)):
        try:
            return json.dumps(raw_result, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(raw_result)
    return str(raw_result)


async def execute_registered_tool(
    spec: LLMSpec,
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
    token: Optional[CancellationToken] = None,
) -> str:
    """
    Pipeline eksekusi tool aman dengan tahapan:
    1. Validasi izin RBAC terhadap spec.tool_permissions.
    2. Verifikasi keberadaan tool di in-memory registry.
    3. Periksa CancellationToken (tunggu jika pause, batalkan jika cancelled).
    4. Validasi argumen pemanggilan terhadap parameters_schema.
    5. Jalankan fungsi handler (asinkron langsung atau sinkron via asyncio.to_thread).
    6. Periksa CancellationToken pasca-eksekusi.
    7. Teruskan hasil keluaran ke Output Spill Guard (batas keras 50 KB).
    """
    clean_name = tool_name.strip()

    # 1. Validasi Izin RBAC
    if not authorize_tool_call(spec, clean_name):
        raise ToolPermissionError(
            f"Agen '{spec.handle}' tidak memiliki hak akses RBAC untuk memanggil tool '{clean_name}'."
        )

    # 2. Verifikasi Keberadaan Tool di Registry
    tool = get_registered_tool(clean_name)
    if tool is None:
        raise ToolNotFoundError(
            f"Tool '{clean_name}' tidak terdaftar pada in-memory permission registry."
        )

    # 3. Periksa CancellationToken Pra-Eksekusi
    if token is not None:
        await token.wait_if_paused()
        token.raise_if_cancelled()

    # 4. Validasi Argumen
    args = arguments if arguments is not None else {}
    if not isinstance(args, dict):
        raise ToolArgumentError(
            f"Argumen untuk tool '{clean_name}' harus berupa dictionary, didapat: {type(args).__name__}."
        )
    _validate_tool_arguments(tool, args)

    # 5. Eksekusi Handler
    try:
        if inspect.iscoroutinefunction(tool.handler):
            raw_result = await tool.handler(**args)
        else:
            raw_result = await asyncio.to_thread(tool.handler, **args)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error(f"[PermissionRegistry] Handler '{clean_name}' gagal dieksekusi: {exc}", exc_info=True)
        raise ToolExecutionError(
            f"Gagal mengeksekusi handler untuk tool '{clean_name}': {exc}"
        ) from exc

    # 6. Periksa CancellationToken Pasca-Eksekusi
    if token is not None:
        token.raise_if_cancelled()

    # 7. Teruskan Hasil ke Output Spill Guard (50KB Bound)
    formatted_output = _format_raw_output(raw_result)
    guarded_output = await guard_tool_output(
        raw_output=formatted_output,
        session_id=spec.handle,
    )

    return guarded_output


__all__ = [
    "ToolSpec",
    "ToolRegistryError",
    "ToolPermissionError",
    "ToolNotFoundError",
    "ToolArgumentError",
    "ToolExecutionError",
    "register_tool",
    "get_registered_tool",
    "list_registered_tools",
    "unregister_tool",
    "clear_tool_registry",
    "authorize_tool_call",
    "export_permitted_tools",
    "execute_registered_tool",
]
