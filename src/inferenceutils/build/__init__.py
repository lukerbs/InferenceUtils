"""
Build configuration utilities for LLM inference engines.

Provides optimal build arguments and install commands.
"""

from .llama_cpp_env import (
    llama_cpp_args,
    install_command,
)

__all__ = [
    "llama_cpp_args",
    "install_command",
]

