"""
LLM Gateway Adapters for LegionADK.
"""

from LegionADK.gateway.adapter.base_adapter import BaseLLMAdapter
from LegionADK.gateway.adapter.openai_compatible import OpenAICompatibleAdapter
from LegionADK.gateway.adapter.anthropic_adapter import AnthropicAdapter
from LegionADK.gateway.adapter.gemini_adapter import GeminiAdapter
from LegionADK.gateway.adapter.custom_adapter import CustomAdapter, CustomProviderAdapter

__all__ = [
    "BaseLLMAdapter",
    "OpenAICompatibleAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "CustomAdapter",
    "CustomProviderAdapter",
]
