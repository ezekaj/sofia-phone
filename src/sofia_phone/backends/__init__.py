"""
Voice Backend Implementations

This package contains various voice backend implementations
for sofia-phone, including memory-enabled and mock backends.
"""

from .memory_voice_backend import MemoryEnabledVoiceBackend

__all__ = ['MemoryEnabledVoiceBackend']
