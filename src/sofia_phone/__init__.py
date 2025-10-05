"""
Sofia Phone - Generic FreeSWITCH Telephony Integration

A reusable phone system infrastructure that works with any voice AI backend.

Example usage:
    from sofia_phone import PhoneHandler
    from my_ai import MyVoiceBackend

    backend = MyVoiceBackend()
    phone = PhoneHandler(voice_backend=backend)
    await phone.start()
"""

__version__ = "0.1.0"

from .interfaces.voice_backend import VoiceBackend

__all__ = ["VoiceBackend"]
