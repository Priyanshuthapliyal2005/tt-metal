"""
Compatibility shim for Ministral-8B model integration.
This module re-exports the necessary components from the tt-transformers framework.
"""

from models.tt_transformers.mistral8b.model import (
    TtTransformer,
    MistralTransformer,
    MistralModelArgs,
    TtModelArgs,
    create_ministral_model,
)

__all__ = [
    'TtTransformer',
    'MistralTransformer',
    'MistralModelArgs',
    'TtModelArgs',
    'create_ministral_model',
]
