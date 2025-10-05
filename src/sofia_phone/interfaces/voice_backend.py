"""
VoiceBackend Interface

This is the contract that ANY AI backend must implement to work with sofia-phone.
Examples: Hotel receptionist, restaurant booking, clinic scheduler, etc.

Critical: This interface handles audio format conversions automatically.
Your backend receives/sends audio at whatever sample rate it needs.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class VoiceBackend(ABC):
    """
    Abstract interface for voice AI backends.

    Any AI system (Sofia, ChatGPT Voice, custom LLM, etc.) must implement
    these three methods to work with sofia-phone telephony infrastructure.
    """

    @abstractmethod
    async def transcribe(self, audio: bytes, sample_rate: int = 8000) -> str:
        """
        Convert audio to text (Speech-to-Text)

        Args:
            audio: Raw PCM audio bytes (mono, 16-bit)
            sample_rate: Sample rate in Hz (default 8000 for telephony)

        Returns:
            Transcribed text string

        Example:
            audio = b'\\x00\\x01...'  # Audio from caller
            text = await backend.transcribe(audio, sample_rate=8000)
            # text = "I need to book a room"
        """
        pass

    @abstractmethod
    async def generate(self, text: str, session_id: str, history: Optional[List[Dict]] = None) -> str:
        """
        Generate AI response from user input

        Args:
            text: Transcribed user speech
            session_id: Unique identifier for this call session
            history: Previous conversation messages for context
                     Format: [{"role": "user", "content": "..."}, ...]

        Returns:
            AI-generated response text

        Example:
            response = await backend.generate(
                text="I need a room for tonight",
                session_id="call-12345",
                history=[...]
            )
            # response = "Certainly! We have rooms available. For how many guests?"
        """
        pass

    @abstractmethod
    async def speak(self, text: str) -> tuple[bytes, int]:
        """
        Convert text to speech (Text-to-Speech)

        Args:
            text: Text to convert to speech

        Returns:
            Tuple of (audio_bytes, sample_rate)
            - audio_bytes: Raw PCM audio (mono, 16-bit)
            - sample_rate: Sample rate of the generated audio (e.g., 24000, 16000)

        Note: sofia-phone will automatically resample to 8kHz for telephony

        Example:
            audio, rate = await backend.speak("How can I help you?")
            # audio = b'\\x00\\x01...'
            # rate = 24000  (Edge TTS outputs 24kHz, sofia-phone resamples to 8kHz)
        """
        pass

    async def on_call_start(self, session_id: str, caller_id: str) -> Optional[str]:
        """
        Optional: Called when a new call starts

        Args:
            session_id: Unique call session identifier
            caller_id: Caller's phone number

        Returns:
            Optional greeting message to speak first

        Example:
            greeting = await backend.on_call_start("call-123", "+1234567890")
            # greeting = "Good morning! Hotel Larchmont, Sofia speaking. How may I help you?"
        """
        return None

    async def on_call_end(self, session_id: str, duration: float) -> None:
        """
        Optional: Called when call ends (cleanup, logging, etc.)

        Args:
            session_id: Call session identifier
            duration: Call duration in seconds
        """
        pass

    async def on_error(self, session_id: str, error: Exception) -> Optional[str]:
        """
        Optional: Handle errors gracefully

        Args:
            session_id: Call session identifier
            error: Exception that occurred

        Returns:
            Optional fallback message to speak to caller

        Example:
            msg = await backend.on_error("call-123", TimeoutError())
            # msg = "I'm sorry, I'm having technical difficulties.
            #        Let me transfer you to a staff member."
        """
        return "I apologize, I'm experiencing technical difficulties. Please hold."
