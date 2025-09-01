"""
inference_wrappers

Este pacote fornece classes Python de alto nível para interagir com
motores de inferência baseados em C++, como Llama.cpp e Whisper.cpp.

Classes disponíveis:
- LlamaCpp: Para geração de texto com modelos de linguagem grandes (formato GGUF).
- WhisperCpp: Para transcrição de áudio para texto (ASR).
"""

from .llama_wrapper import LlamaCpp
from .whisper_wrapper import WhisperCpp

__all__ = [
    "LlamaCpp",
    "WhisperCpp"
]
