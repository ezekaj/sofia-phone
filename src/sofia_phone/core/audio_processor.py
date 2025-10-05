"""
Production-Grade Audio Processor

Handles all audio format conversions for telephony integration.
Critical for maintaining call quality across different sample rates.

Features:
- Professional resampling (scipy.signal.resample_poly)
- Format validation and normalization
- Audio quality metrics
- Buffer management
- Error recovery
"""
import numpy as np
from scipy import signal
from typing import Tuple, Optional
from dataclasses import dataclass
import struct
from loguru import logger


@dataclass
class AudioMetrics:
    """Audio quality metrics for monitoring"""
    amplitude_max: int
    amplitude_mean: float
    sample_rate: int
    duration_ms: float
    clipping_detected: bool
    silence_detected: bool

    def __str__(self) -> str:
        return (
            f"AudioMetrics(max={self.amplitude_max}, mean={self.amplitude_mean:.1f}, "
            f"rate={self.sample_rate}Hz, duration={self.duration_ms:.0f}ms, "
            f"clipping={'YES' if self.clipping_detected else 'NO'}, "
            f"silence={'YES' if self.silence_detected else 'NO'})"
        )


class AudioProcessor:
    """
    Production-grade audio processing for telephony.

    Handles conversions between:
    - FreeSWITCH (8kHz, mono, 16-bit PCM)
    - Whisper STT (16kHz, mono, 16-bit PCM)
    - Edge TTS (24kHz, mono, 16-bit PCM)

    All conversions use professional-grade resampling to maintain quality.
    """

    # Telephony standards
    TELEPHONY_SAMPLE_RATE = 8000  # 8kHz (G.711 standard)
    WHISPER_SAMPLE_RATE = 16000   # 16kHz (Whisper optimal)
    TTS_SAMPLE_RATE = 24000       # 24kHz (Edge TTS default)

    # Audio quality thresholds
    SILENCE_THRESHOLD = 500       # Amplitude below this = silence
    CLIPPING_THRESHOLD = 30000    # Amplitude above this = clipping (max 32767)
    MIN_AUDIO_DURATION_MS = 100   # Minimum viable audio length

    def __init__(self):
        """Initialize audio processor"""
        logger.info("AudioProcessor initialized")

    def resample(
        self,
        audio: bytes,
        from_rate: int,
        to_rate: int,
        validate: bool = True
    ) -> bytes:
        """
        Resample audio with professional quality.

        Uses scipy.signal.resample_poly for high-quality resampling
        with minimal artifacts.

        Args:
            audio: Raw PCM audio bytes (mono, 16-bit)
            from_rate: Source sample rate (Hz)
            to_rate: Target sample rate (Hz)
            validate: Whether to validate audio quality

        Returns:
            Resampled audio bytes (mono, 16-bit PCM)

        Raises:
            ValueError: If audio is invalid or empty
            RuntimeError: If resampling fails

        Example:
            # 8kHz → 16kHz for Whisper
            whisper_audio = processor.resample(telephony_audio, 8000, 16000)

            # 24kHz → 8kHz for telephony
            telephony_audio = processor.resample(tts_audio, 24000, 8000)
        """
        try:
            # Validate input
            if not audio:
                raise ValueError("Empty audio buffer")

            if len(audio) % 2 != 0:
                raise ValueError("Audio buffer must be even length (16-bit samples)")

            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio, dtype=np.int16)

            # Validate minimum duration
            duration_ms = (len(audio_array) / from_rate) * 1000
            if duration_ms < self.MIN_AUDIO_DURATION_MS:
                logger.warning(
                    f"Short audio: {duration_ms:.1f}ms (min {self.MIN_AUDIO_DURATION_MS}ms)"
                )

            # No resampling needed if rates match
            if from_rate == to_rate:
                if validate:
                    metrics = self._analyze_audio(audio_array, from_rate)
                    logger.debug(f"Audio passthrough: {metrics}")
                return audio

            # Calculate resampling ratio
            # Use greatest common divisor for optimal quality
            from typing import Tuple as TupleType
            gcd = np.gcd(from_rate, to_rate)
            up = to_rate // gcd
            down = from_rate // gcd

            logger.debug(
                f"Resampling: {from_rate}Hz → {to_rate}Hz "
                f"(up={up}, down={down}, gcd={gcd})"
            )

            # Perform high-quality resampling
            # resample_poly uses polyphase filtering for minimal artifacts
            resampled = signal.resample_poly(audio_array, up, down)

            # Ensure output is int16
            # Clip to prevent overflow
            resampled = np.clip(resampled, -32768, 32767).astype(np.int16)

            # Validate output quality
            if validate:
                input_metrics = self._analyze_audio(audio_array, from_rate)
                output_metrics = self._analyze_audio(resampled, to_rate)

                logger.debug(f"Input:  {input_metrics}")
                logger.debug(f"Output: {output_metrics}")

                # Warn if quality degraded
                if output_metrics.clipping_detected and not input_metrics.clipping_detected:
                    logger.warning("Clipping introduced during resampling")

            return resampled.tobytes()

        except Exception as e:
            logger.error(f"Resampling failed ({from_rate}→{to_rate}): {e}")
            raise RuntimeError(f"Audio resampling error: {e}") from e

    def _analyze_audio(self, audio_array: np.ndarray, sample_rate: int) -> AudioMetrics:
        """
        Analyze audio quality metrics.

        Args:
            audio_array: Audio samples as numpy array
            sample_rate: Sample rate in Hz

        Returns:
            AudioMetrics with quality information
        """
        amplitude_max = int(np.abs(audio_array).max())
        amplitude_mean = float(np.abs(audio_array).mean())
        duration_ms = (len(audio_array) / sample_rate) * 1000

        # Detect issues
        clipping_detected = amplitude_max >= self.CLIPPING_THRESHOLD
        silence_detected = amplitude_max < self.SILENCE_THRESHOLD

        return AudioMetrics(
            amplitude_max=amplitude_max,
            amplitude_mean=amplitude_mean,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            clipping_detected=clipping_detected,
            silence_detected=silence_detected
        )

    def normalize_volume(self, audio: bytes, sample_rate: int, target_peak: float = 0.8) -> bytes:
        """
        Normalize audio volume to target peak level.

        Prevents audio from being too quiet or clipping.

        Args:
            audio: Raw PCM audio bytes
            sample_rate: Sample rate in Hz
            target_peak: Target peak level (0.0-1.0), default 0.8

        Returns:
            Volume-normalized audio bytes
        """
        audio_array = np.frombuffer(audio, dtype=np.int16)

        # Find current peak (as fraction of max int16)
        current_peak = np.abs(audio_array).max() / 32767.0

        if current_peak == 0:
            logger.warning("Silent audio, cannot normalize")
            return audio

        # Calculate gain
        gain = target_peak / current_peak

        # Apply gain with clipping protection
        normalized = audio_array * gain
        normalized = np.clip(normalized, -32768, 32767).astype(np.int16)

        logger.debug(f"Volume normalized: peak {current_peak:.2f} → {target_peak:.2f} (gain={gain:.2f}x)")

        return normalized.tobytes()

    def detect_voice_activity(self, audio: bytes, sample_rate: int) -> bool:
        """
        Simple voice activity detection.

        Args:
            audio: Raw PCM audio bytes
            sample_rate: Sample rate in Hz

        Returns:
            True if voice activity detected
        """
        audio_array = np.frombuffer(audio, dtype=np.int16)
        amplitude_max = np.abs(audio_array).max()

        return amplitude_max >= self.SILENCE_THRESHOLD

    def get_duration_ms(self, audio: bytes, sample_rate: int) -> float:
        """
        Calculate audio duration in milliseconds.

        Args:
            audio: Raw PCM audio bytes (16-bit)
            sample_rate: Sample rate in Hz

        Returns:
            Duration in milliseconds
        """
        num_samples = len(audio) // 2  # 2 bytes per 16-bit sample
        return (num_samples / sample_rate) * 1000.0

    def split_by_silence(
        self,
        audio: bytes,
        sample_rate: int,
        silence_threshold: int = 500,
        min_silence_ms: int = 300
    ) -> list[bytes]:
        """
        Split audio at silence boundaries.

        Useful for breaking long audio into sentences.

        Args:
            audio: Raw PCM audio bytes
            sample_rate: Sample rate in Hz
            silence_threshold: Amplitude threshold for silence
            min_silence_ms: Minimum silence duration to split

        Returns:
            List of audio chunks
        """
        audio_array = np.frombuffer(audio, dtype=np.int16)
        min_silence_samples = int((min_silence_ms / 1000.0) * sample_rate)

        chunks = []
        current_chunk = []
        silence_count = 0

        for sample in audio_array:
            amplitude = abs(sample)

            if amplitude < silence_threshold:
                silence_count += 1
                current_chunk.append(sample)

                # Found silence boundary
                if silence_count >= min_silence_samples and current_chunk:
                    chunks.append(np.array(current_chunk, dtype=np.int16).tobytes())
                    current_chunk = []
                    silence_count = 0
            else:
                current_chunk.append(sample)
                silence_count = 0

        # Add final chunk
        if current_chunk:
            chunks.append(np.array(current_chunk, dtype=np.int16).tobytes())

        logger.debug(f"Split audio into {len(chunks)} chunks at silence boundaries")
        return chunks
