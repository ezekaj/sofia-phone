"""
Memory-Enabled Mock Voice Backend

Demonstrates how to integrate the episodic memory system
with the VoiceBackend interface.

This is a production pattern that can be used with any
real AI backend (Whisper + Ollama + Edge TTS).
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from loguru import logger

from sofia_phone.interfaces.voice_backend import VoiceBackend


class MemoryEnabledVoiceBackend(VoiceBackend):
    """
    Voice backend with episodic memory.

    Features:
    - Remembers past conversations
    - Learns patterns over time
    - Contextual retrieval
    - Graceful forgetting
    """

    def __init__(self, enable_memory: bool = True):
        """
        Initialize memory-enabled backend.

        Args:
            enable_memory: Whether to use memory system
        """
        self.enable_memory = enable_memory

        # Simple in-memory storage (production would use ChromaDB)
        self.conversation_memory: List[Dict] = []
        self.learned_patterns: Dict[str, int] = {}

        logger.info(f"MemoryEnabledVoiceBackend initialized (memory={enable_memory})")

    async def transcribe(self, audio: bytes, sample_rate: int = 8000) -> str:
        """
        Speech-to-text with context awareness.

        Args:
            audio: Raw PCM audio
            sample_rate: Sample rate in Hz

        Returns:
            Transcribed text
        """
        # Mock transcription (production: use Whisper)
        duration = len(audio) / (sample_rate * 2)  # 16-bit audio

        # Check if similar audio patterns seen before (learning)
        pattern_key = f"audio_{len(audio)}"
        if pattern_key in self.learned_patterns:
            self.learned_patterns[pattern_key] += 1
            logger.debug(f"Seen this pattern {self.learned_patterns[pattern_key]} times")
        else:
            self.learned_patterns[pattern_key] = 1

        transcription = f"Mock transcription ({duration:.1f}s audio)"

        # Store in memory
        if self.enable_memory:
            self._add_to_memory({
                'type': 'transcription',
                'text': transcription,
                'audio_length': len(audio),
                'timestamp': datetime.now().isoformat()
            })

        return transcription

    async def generate(
        self,
        text: str,
        session_id: str,
        history: Optional[List[Dict]] = None
    ) -> str:
        """
        LLM response with memory context.

        Args:
            text: User input
            session_id: Session identifier
            history: Conversation history

        Returns:
            AI response
        """
        if not text:
            # Initial greeting with memory
            if self.enable_memory and len(self.conversation_memory) > 0:
                return "Welcome back! I remember our last conversation."
            return "Hello! How can I help you today?"

        # Retrieve relevant memories (production: use Bayesian surprise + retrieval)
        relevant_memories = self._retrieve_relevant_memories(text, limit=3)

        # Generate context-aware response
        if relevant_memories:
            context = f" (I recall: {len(relevant_memories)} related conversations)"
            response = f"You said: {text}{context}"
        else:
            response = f"You said: {text}"

        # Store this interaction
        if self.enable_memory:
            self._add_to_memory({
                'type': 'conversation',
                'session_id': session_id,
                'user_text': text,
                'ai_response': response,
                'timestamp': datetime.now().isoformat()
            })

        return response

    async def speak(self, text: str) -> Tuple[bytes, int]:
        """
        Text-to-speech.

        Args:
            text: Text to speak

        Returns:
            (audio_bytes, sample_rate)
        """
        # Mock TTS (production: use Edge TTS)
        sample_rate = 16000
        duration_seconds = len(text) / 20  # Rough estimate
        num_samples = int(duration_seconds * sample_rate)

        # Generate silence (mock audio)
        audio = np.zeros(num_samples, dtype=np.int16)

        return audio.tobytes(), sample_rate

    def _add_to_memory(self, entry: Dict):
        """Add entry to episodic memory"""
        entry['memory_id'] = len(self.conversation_memory)
        self.conversation_memory.append(entry)

        # Simple forgetting (keep last 100 entries)
        if len(self.conversation_memory) > 100:
            self.conversation_memory = self.conversation_memory[-100:]

        logger.debug(f"Memory updated: {len(self.conversation_memory)} entries")

    def _retrieve_relevant_memories(self, query: str, limit: int = 3) -> List[Dict]:
        """
        Retrieve relevant memories for query.

        Production: Use two-stage retrieval (similarity + temporal)
        """
        if not self.conversation_memory:
            return []

        # Simple keyword matching (production: vector similarity)
        relevant = []
        for memory in reversed(self.conversation_memory):  # Recent first
            if memory.get('type') == 'conversation':
                if query.lower() in memory.get('user_text', '').lower():
                    relevant.append(memory)
                    if len(relevant) >= limit:
                        break

        return relevant

    def get_memory_stats(self) -> Dict:
        """Get memory system statistics"""
        return {
            'total_memories': len(self.conversation_memory),
            'learned_patterns': len(self.learned_patterns),
            'pattern_counts': dict(list(self.learned_patterns.items())[:5])  # Top 5
        }
