"""
audio_processing

Este pacote contém a lógica para o pipeline de áudio em tempo real,
incluindo detecção de wake word, VAD e síntese de voz.
"""
from .pipeline import AudioOrchestrator

__all__ = ["AudioOrchestrator"]
