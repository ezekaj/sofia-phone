"""
Mock Voice Backend for Testing

A simple echo bot that implements VoiceBackend interface.
Useful for testing sofia-phone without real AI dependencies.
"""
import sys
sys.path.insert(0, '../../src')

from sofia_phone.interfaces.voice_backend import VoiceBackend
from typing import List, Dict, Optional
import numpy as np


class MockVoiceBackend(VoiceBackend):
    """
    Simple echo bot for testing.

    - STT: Returns dummy text
    - LLM: Echoes back what was said
    - TTS: Generates silent audio
    """

    async def transcribe(self, audio: bytes, sample_rate: int = 8000) -> str:
        """Mock STT - returns dummy text"""
        # In real testing, could analyze audio properties
        audio_array = np.frombuffer(audio, dtype=np.int16)
        duration = len(audio_array) / sample_rate

        return f"Mock transcription ({duration:.1f}s audio)"

    async def generate(self, text: str, session_id: str, history: Optional[List[Dict]] = None) -> str:
        """Mock LLM - simple echo"""
        return f"You said: {text}"

    async def speak(self, text: str) -> tuple[bytes, int]:
        """Mock TTS - generates 1 second of silence"""
        sample_rate = 16000
        duration = 1.0  # 1 second
        num_samples = int(sample_rate * duration)

        # Generate silence (zeros)
        audio = np.zeros(num_samples, dtype=np.int16)

        return audio.tobytes(), sample_rate

    async def on_call_start(self, session_id: str, caller_id: str) -> Optional[str]:
        """Greeting message"""
        return f"Hello! This is a test echo bot. Session {session_id[:8]}."

    async def on_call_end(self, session_id: str, duration: float) -> None:
        """Log call end"""
        print(f"[MockBackend] Call {session_id} ended after {duration:.1f}s")

    async def on_error(self, session_id: str, error: Exception) -> Optional[str]:
        """Handle errors"""
        print(f"[MockBackend] Error in {session_id}: {error}")
        return "Test error occurred"
