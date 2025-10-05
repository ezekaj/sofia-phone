"""
Production-Grade Phone Handler (Main Orchestrator)

This is the core component that ties everything together:
- Accepts calls via FreeSWITCH ESL (Event Socket Layer)
- Manages call sessions
- Routes audio through RTP handler
- Integrates with VoiceBackend (STT/LLM/TTS)
- Handles interruptions
- Production error recovery

Architecture:
┌─────────────┐     ESL      ┌──────────────┐
│ FreeSWITCH  │◄────────────►│ PhoneHandler │
│             │     RTP      │              │
│  (port 8084)│◄────────────►│              │
└─────────────┘              └──────┬───────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
              │  Session  │  │    RTP    │  │   Voice   │
              │  Manager  │  │  Handler  │  │  Backend  │
              └───────────┘  └───────────┘  └───────────┘
"""
import asyncio
from typing import Optional, Dict, Any
from loguru import logger
from collections import deque
import time

from ..interfaces.voice_backend import VoiceBackend
from .session_manager import SessionManager, SessionState
from .audio_processor import AudioProcessor
from ..freeswitch.rtp_handler import RTPHandler


class CallHandler:
    """
    Handles a single active call.

    Created per call - manages audio streaming, transcription,
    LLM conversation, TTS, and interruption handling.
    """

    def __init__(
        self,
        session_id: str,
        caller_id: str,
        voice_backend: VoiceBackend,
        audio_processor: AudioProcessor,
        rtp_handler: RTPHandler,
        session_manager: SessionManager
    ):
        """
        Initialize call handler.

        Args:
            session_id: Unique session identifier
            caller_id: Caller's phone number
            voice_backend: AI backend (STT/LLM/TTS)
            audio_processor: Audio resampling
            rtp_handler: RTP audio streaming
            session_manager: Session management
        """
        self.session_id = session_id
        self.caller_id = caller_id
        self.voice_backend = voice_backend
        self.audio_processor = audio_processor
        self.rtp_handler = rtp_handler
        self.session_manager = session_manager

        # Call state
        self.active = False
        self.is_speaking = False
        self.remote_rtp_host: Optional[str] = None
        self.remote_rtp_port: Optional[int] = None

        # Audio buffering for STT
        # Buffer incoming audio until we have enough for transcription
        self.audio_buffer = deque(maxlen=100)  # Max 2 seconds at 20ms chunks
        self.buffer_duration_ms = 0
        self.silence_duration_ms = 0

        # Interruption handling
        self.interrupt_speech_count = 0
        self.INTERRUPT_THRESHOLD = 5  # Require sustained speech (100ms)

        # Configuration
        self.MIN_TRANSCRIPTION_MS = 500  # Minimum audio for transcription
        self.MAX_SILENCE_MS = 800  # Silence threshold to trigger transcription
        self.CHUNK_DURATION_MS = 20  # FreeSWITCH sends 20ms chunks at 8kHz

        logger.info(f"CallHandler initialized for {caller_id} (session={session_id[:8]})")

    async def start(self, remote_rtp_host: str, remote_rtp_port: int):
        """
        Start call handling.

        Args:
            remote_rtp_host: FreeSWITCH RTP host
            remote_rtp_port: FreeSWITCH RTP port
        """
        self.active = True
        self.remote_rtp_host = remote_rtp_host
        self.remote_rtp_port = remote_rtp_port

        # Update session state
        self.session_manager.update_session_state(self.session_id, SessionState.ACTIVE)

        logger.info(
            f"Call started: {self.caller_id} → RTP {remote_rtp_host}:{remote_rtp_port}"
        )

        # Send initial greeting
        await self._send_greeting()

    async def _send_greeting(self):
        """Send initial greeting to caller"""
        try:
            # Get greeting from LLM
            greeting = await self.voice_backend.generate(
                text="",  # Empty = initial greeting
                session_id=self.session_id,
                history=[]
            )

            logger.info(f"Greeting: {greeting}")

            # Convert to speech and send
            await self._speak(greeting)

        except Exception as e:
            logger.error(f"Failed to send greeting: {e}")
            # Fallback to simple greeting
            await self._speak("Hello, how can I help you today?")

    async def handle_incoming_audio(self, audio: bytes):
        """
        Handle incoming RTP audio from FreeSWITCH.

        This is called by RTP handler for each audio chunk (20ms at 8kHz).
        Buffers audio and triggers transcription when appropriate.

        Args:
            audio: Raw PCM audio (8kHz, mono, 16-bit)
        """
        if not self.active:
            return

        try:
            # Check if this is voice activity
            has_voice = self.audio_processor.detect_voice_activity(
                audio,
                AudioProcessor.TELEPHONY_SAMPLE_RATE
            )

            # Handle interruption detection
            if self.is_speaking and has_voice:
                self.interrupt_speech_count += 1

                if self.interrupt_speech_count >= self.INTERRUPT_THRESHOLD:
                    logger.info("User interruption detected - stopping TTS")
                    self.is_speaking = False
                    self.interrupt_speech_count = 0
                    # TODO: Cancel ongoing TTS playback

            # Buffer audio
            self.audio_buffer.append(audio)
            self.buffer_duration_ms += self.CHUNK_DURATION_MS

            if has_voice:
                self.silence_duration_ms = 0
            else:
                self.silence_duration_ms += self.CHUNK_DURATION_MS

            # Trigger transcription if:
            # 1. We have minimum audio duration AND
            # 2. User stopped talking (silence threshold reached)
            should_transcribe = (
                self.buffer_duration_ms >= self.MIN_TRANSCRIPTION_MS and
                self.silence_duration_ms >= self.MAX_SILENCE_MS
            )

            if should_transcribe and len(self.audio_buffer) > 0:
                # Combine buffered audio
                combined_audio = b''.join(self.audio_buffer)

                # Clear buffer
                self.audio_buffer.clear()
                self.buffer_duration_ms = 0
                self.silence_duration_ms = 0

                # Process in background (don't block RTP)
                asyncio.create_task(self._process_user_speech(combined_audio))

        except Exception as e:
            logger.error(f"Error handling incoming audio: {e}")
            session = self.session_manager.get_session(self.session_id)
            if session:
                session.errors_count += 1

    async def _process_user_speech(self, audio: bytes):
        """
        Process user speech: STT → LLM → TTS → RTP

        This is the main conversational flow.

        Args:
            audio: Raw PCM audio (8kHz, mono, 16-bit)
        """
        try:
            session = self.session_manager.get_session(self.session_id)
            if not session:
                logger.error(f"Session {self.session_id} not found")
                return

            # 1. Resample for Whisper (8kHz → 16kHz)
            whisper_audio = self.audio_processor.resample(
                audio,
                from_rate=AudioProcessor.TELEPHONY_SAMPLE_RATE,
                to_rate=AudioProcessor.WHISPER_SAMPLE_RATE
            )

            logger.debug(f"Resampled audio: 8kHz → 16kHz ({len(audio)} → {len(whisper_audio)} bytes)")

            # 2. Transcribe
            transcription = await self.voice_backend.transcribe(
                whisper_audio,
                sample_rate=AudioProcessor.WHISPER_SAMPLE_RATE
            )

            if not transcription or len(transcription.strip()) == 0:
                logger.debug("Empty transcription - ignoring")
                return

            logger.info(f"User: {transcription}")
            session.transcriptions_count += 1

            # 3. Generate LLM response
            response = await self.voice_backend.generate(
                text=transcription,
                session_id=self.session_id,
                history=session.conversation_history
            )

            logger.info(f"AI: {response}")
            session.llm_responses_count += 1

            # 4. Update conversation history
            session.add_conversation_turn(transcription, response)

            # 5. Convert to speech and send
            await self._speak(response)

        except Exception as e:
            logger.error(f"Error processing user speech: {e}")
            session = self.session_manager.get_session(self.session_id)
            if session:
                session.errors_count += 1

            # Send error message to user
            await self._speak("I'm sorry, I didn't catch that. Could you please repeat?")

    async def _speak(self, text: str):
        """
        Convert text to speech and send via RTP.

        Args:
            text: Text to speak
        """
        try:
            self.is_speaking = True

            # 1. Generate TTS audio
            tts_audio, tts_sample_rate = await self.voice_backend.speak(text)

            logger.debug(f"TTS generated: {len(tts_audio)} bytes at {tts_sample_rate}Hz")

            # 2. Resample to telephony rate (TTS rate → 8kHz)
            telephony_audio = self.audio_processor.resample(
                tts_audio,
                from_rate=tts_sample_rate,
                to_rate=AudioProcessor.TELEPHONY_SAMPLE_RATE
            )

            logger.debug(f"Resampled TTS: {tts_sample_rate}Hz → 8kHz ({len(tts_audio)} → {len(telephony_audio)} bytes)")

            # 3. Send via RTP in chunks
            # FreeSWITCH expects 20ms chunks (160 bytes at 8kHz, 16-bit mono)
            chunk_size = 160 * 2  # 160 samples * 2 bytes per sample = 320 bytes

            for i in range(0, len(telephony_audio), chunk_size):
                if not self.is_speaking:
                    logger.info("TTS interrupted by user")
                    break

                chunk = telephony_audio[i:i + chunk_size]

                # Pad last chunk if needed
                if len(chunk) < chunk_size:
                    chunk += b'\x00' * (chunk_size - len(chunk))

                # Send via RTP
                await self.rtp_handler.send_audio(
                    chunk,
                    self.remote_rtp_host,
                    self.remote_rtp_port
                )

                # Wait 20ms (simulate real-time playback)
                await asyncio.sleep(0.02)

            self.is_speaking = False

            # Update session metrics
            session = self.session_manager.get_session(self.session_id)
            if session:
                session.audio_chunks_sent += len(telephony_audio) // chunk_size

        except Exception as e:
            logger.error(f"Error in TTS: {e}")
            self.is_speaking = False

            session = self.session_manager.get_session(self.session_id)
            if session:
                session.errors_count += 1

    async def stop(self):
        """Stop call handling"""
        self.active = False
        self.is_speaking = False
        self.audio_buffer.clear()

        logger.info(f"Call stopped: {self.caller_id}")


class PhoneHandler:
    """
    Production-grade phone system orchestrator.

    This is the main entry point that:
    - Listens for ESL connections from FreeSWITCH
    - Creates call sessions
    - Manages RTP audio streaming
    - Orchestrates VoiceBackend integration
    - Handles errors and recovery

    Usage:
        handler = PhoneHandler(voice_backend=sofia_backend)
        await handler.start()
    """

    def __init__(
        self,
        voice_backend: VoiceBackend,
        esl_host: str = "0.0.0.0",
        esl_port: int = 8084,
        rtp_start_port: int = 16384,
        max_concurrent_calls: int = 100
    ):
        """
        Initialize phone handler.

        Args:
            voice_backend: AI backend (STT/LLM/TTS)
            esl_host: ESL listen host
            esl_port: ESL listen port (FreeSWITCH connects here)
            rtp_start_port: Starting RTP port for calls
            max_concurrent_calls: Maximum concurrent calls
        """
        self.voice_backend = voice_backend
        self.esl_host = esl_host
        self.esl_port = esl_port
        self.rtp_start_port = rtp_start_port
        self.max_concurrent_calls = max_concurrent_calls

        # Core components
        self.session_manager = SessionManager(max_sessions=max_concurrent_calls)
        self.audio_processor = AudioProcessor()

        # Active calls
        self._active_calls: Dict[str, CallHandler] = {}
        self._next_rtp_port = rtp_start_port

        # ESL server
        self._esl_server: Optional[asyncio.Server] = None
        self._running = False

        logger.info(
            f"PhoneHandler initialized (ESL {esl_host}:{esl_port}, "
            f"RTP ports {rtp_start_port}+, max calls={max_concurrent_calls})"
        )

    async def start(self):
        """Start phone handler"""
        if self._running:
            logger.warning("PhoneHandler already running")
            return

        self._running = True

        # Start session manager
        await self.session_manager.start()

        # Start ESL server
        self._esl_server = await asyncio.start_server(
            self._handle_esl_connection,
            self.esl_host,
            self.esl_port
        )

        logger.success(
            f"PhoneHandler started - listening for calls on "
            f"{self.esl_host}:{self.esl_port}"
        )
        logger.info("Waiting for FreeSWITCH to connect...")

    async def stop(self):
        """Stop phone handler"""
        self._running = False

        # Stop ESL server
        if self._esl_server:
            self._esl_server.close()
            await self._esl_server.wait_closed()

        # Stop all active calls
        for call_handler in list(self._active_calls.values()):
            await call_handler.stop()

        # Stop session manager
        await self.session_manager.stop()

        logger.info("PhoneHandler stopped")

    async def _handle_esl_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ):
        """
        Handle incoming ESL connection from FreeSWITCH.

        This is called when FreeSWITCH executes the socket application.

        Args:
            reader: Stream reader
            writer: Stream writer
        """
        try:
            logger.info("ESL connection received from FreeSWITCH")

            # Read initial event (call info)
            headers = await self._read_esl_headers(reader)

            caller_id = headers.get("Caller-Caller-ID-Number", "unknown")
            unique_id = headers.get("Unique-ID", "unknown")

            logger.info(f"Incoming call from {caller_id} (unique_id={unique_id[:8]})")

            # Create session
            session = self.session_manager.create_session(
                caller_id=caller_id,
                metadata={"unique_id": unique_id, "headers": headers}
            )

            # Allocate RTP port
            rtp_port = self._allocate_rtp_port()

            # Create RTP handler
            rtp_handler = RTPHandler(local_port=rtp_port)

            # Create call handler
            call_handler = CallHandler(
                session_id=session.session_id,
                caller_id=caller_id,
                voice_backend=self.voice_backend,
                audio_processor=self.audio_processor,
                rtp_handler=rtp_handler,
                session_manager=self.session_manager
            )

            # Register audio callback
            await rtp_handler.start(
                audio_callback=call_handler.handle_incoming_audio
            )

            # Store call handler
            self._active_calls[session.session_id] = call_handler

            # Get remote RTP info from FreeSWITCH
            # (In production, parse from SDP in headers)
            remote_rtp_host = "127.0.0.1"  # FreeSWITCH is local
            remote_rtp_port = int(headers.get("variable_rtp_local_sdp_port", "16384"))

            # Start call
            await call_handler.start(remote_rtp_host, remote_rtp_port)

            # Keep connection alive until call ends
            # (In production, parse DTMF, handle hangup events, etc.)
            await self._monitor_call(reader, writer, session.session_id)

        except Exception as e:
            logger.error(f"Error handling ESL connection: {e}")

        finally:
            writer.close()
            await writer.wait_closed()

    async def _read_esl_headers(self, reader: asyncio.StreamReader) -> Dict[str, str]:
        """
        Read ESL event headers.

        Args:
            reader: Stream reader

        Returns:
            Dictionary of headers
        """
        headers = {}

        while True:
            line = await reader.readline()
            line = line.decode().strip()

            if not line:  # Empty line = end of headers
                break

            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip()] = value.strip()

        return headers

    async def _monitor_call(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        session_id: str
    ):
        """
        Monitor call for hangup events.

        Args:
            reader: Stream reader
            writer: Stream writer
            session_id: Session identifier
        """
        try:
            while self._running:
                # Read events from FreeSWITCH
                headers = await asyncio.wait_for(
                    self._read_esl_headers(reader),
                    timeout=1.0
                )

                event_name = headers.get("Event-Name")

                if event_name == "CHANNEL_HANGUP":
                    logger.info(f"Call hangup detected for session {session_id[:8]}")
                    break

        except asyncio.TimeoutError:
            # No events - continue monitoring
            pass
        except Exception as e:
            logger.error(f"Error monitoring call: {e}")

        finally:
            # End call
            await self._end_call(session_id)

    async def _end_call(self, session_id: str):
        """
        End call and cleanup resources.

        Args:
            session_id: Session identifier
        """
        call_handler = self._active_calls.get(session_id)
        if call_handler:
            await call_handler.stop()
            await call_handler.rtp_handler.stop()
            del self._active_calls[session_id]

        await self.session_manager.end_session(session_id)

    def _allocate_rtp_port(self) -> int:
        """Allocate next available RTP port"""
        port = self._next_rtp_port
        self._next_rtp_port += 2  # RTP uses even ports, RTCP uses odd

        # Wrap around if we exceed range
        if self._next_rtp_port > 32768:
            self._next_rtp_port = self.rtp_start_port

        return port

    def get_metrics(self) -> Dict[str, Any]:
        """Get phone handler metrics"""
        return {
            "active_calls": len(self._active_calls),
            "session_manager": self.session_manager.get_metrics(),
            "rtp_handlers": [
                handler.rtp_handler.get_metrics()
                for handler in self._active_calls.values()
            ]
        }

    def __str__(self) -> str:
        return (
            f"PhoneHandler(active_calls={len(self._active_calls)}, "
            f"sessions={self.session_manager.get_session_count()})"
        )
