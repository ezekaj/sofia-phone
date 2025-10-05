"""
Tests for AudioProcessor

Tests all audio processing functionality:
- Resampling between different rates
- Quality validation
- Voice activity detection
- Volume normalization
"""
import pytest
import numpy as np
from sofia_phone.core.audio_processor import AudioProcessor, AudioMetrics


@pytest.fixture
def processor():
    """Audio processor fixture"""
    return AudioProcessor()


class TestResampling:
    """Test audio resampling"""

    def test_8k_to_16k(self, processor):
        """Test 8kHz → 16kHz resampling (telephony → whisper)"""
        # Generate 1 second of test audio at 8kHz
        duration_seconds = 1.0
        sample_rate = 8000
        samples = int(duration_seconds * sample_rate)

        # Generate sine wave (1kHz tone)
        audio = np.sin(2 * np.pi * 1000 * np.arange(samples) / sample_rate)
        audio = (audio * 10000).astype(np.int16)
        audio_bytes = audio.tobytes()

        # Resample to 16kHz
        resampled = processor.resample(
            audio_bytes,
            from_rate=8000,
            to_rate=16000
        )

        # Check output length (should be 2x)
        expected_samples = samples * 2
        actual_samples = len(resampled) // 2  # 2 bytes per sample

        assert abs(actual_samples - expected_samples) < 10, \
            f"Expected ~{expected_samples} samples, got {actual_samples}"

    def test_24k_to_8k(self, processor):
        """Test 24kHz → 8kHz resampling (TTS → telephony)"""
        # Generate 1 second of test audio at 24kHz
        duration_seconds = 1.0
        sample_rate = 24000
        samples = int(duration_seconds * sample_rate)

        audio = np.sin(2 * np.pi * 1000 * np.arange(samples) / sample_rate)
        audio = (audio * 10000).astype(np.int16)
        audio_bytes = audio.tobytes()

        # Resample to 8kHz
        resampled = processor.resample(
            audio_bytes,
            from_rate=24000,
            to_rate=8000
        )

        # Check output length (should be 1/3)
        expected_samples = samples // 3
        actual_samples = len(resampled) // 2

        assert abs(actual_samples - expected_samples) < 10, \
            f"Expected ~{expected_samples} samples, got {actual_samples}"

    def test_same_rate_passthrough(self, processor):
        """Test that same rate returns original audio"""
        audio = np.random.randint(-1000, 1000, 1600, dtype=np.int16).tobytes()

        resampled = processor.resample(
            audio,
            from_rate=8000,
            to_rate=8000
        )

        assert audio == resampled, "Same rate should return original audio"

    def test_empty_audio_raises(self, processor):
        """Test that empty audio raises ValueError"""
        with pytest.raises(ValueError, match="Empty audio"):
            processor.resample(b"", from_rate=8000, to_rate=16000)

    def test_odd_length_raises(self, processor):
        """Test that odd-length audio raises ValueError"""
        with pytest.raises(ValueError, match="even length"):
            processor.resample(b"abc", from_rate=8000, to_rate=16000)


class TestQualityMetrics:
    """Test audio quality analysis"""

    def test_silence_detection(self, processor):
        """Test silence detection"""
        # Generate very quiet audio
        audio = np.random.randint(-100, 100, 1600, dtype=np.int16)

        metrics = processor._analyze_audio(audio, 8000)

        assert metrics.silence_detected, "Should detect silence"
        assert metrics.amplitude_max < 500

    def test_clipping_detection(self, processor):
        """Test clipping detection"""
        # Generate audio near max amplitude
        audio = np.full(1600, 32000, dtype=np.int16)

        metrics = processor._analyze_audio(audio, 8000)

        assert metrics.clipping_detected, "Should detect clipping"
        assert metrics.amplitude_max >= 30000

    def test_duration_calculation(self, processor):
        """Test duration calculation"""
        # 1 second at 8kHz = 8000 samples
        audio = np.zeros(8000, dtype=np.int16)

        metrics = processor._analyze_audio(audio, 8000)

        assert abs(metrics.duration_ms - 1000) < 1, \
            f"Expected 1000ms, got {metrics.duration_ms}ms"


class TestVoiceActivity:
    """Test voice activity detection"""

    def test_voice_detected(self, processor):
        """Test voice activity detection (active)"""
        # Generate audio above threshold
        audio = np.random.randint(-5000, 5000, 1600, dtype=np.int16).tobytes()

        has_voice = processor.detect_voice_activity(audio, 8000)

        assert has_voice, "Should detect voice activity"

    def test_silence_detected(self, processor):
        """Test voice activity detection (silent)"""
        # Generate quiet audio
        audio = np.random.randint(-100, 100, 1600, dtype=np.int16).tobytes()

        has_voice = processor.detect_voice_activity(audio, 8000)

        assert not has_voice, "Should detect silence"


class TestVolumeNormalization:
    """Test volume normalization"""

    def test_normalize_quiet_audio(self, processor):
        """Test normalizing quiet audio"""
        # Generate quiet audio (peak = 5000)
        audio = np.random.randint(-5000, 5000, 8000, dtype=np.int16).tobytes()

        normalized = processor.normalize_volume(audio, 8000, target_peak=0.8)

        # Check that peak is now ~80% of max (0.8 * 32767)
        normalized_array = np.frombuffer(normalized, dtype=np.int16)
        peak = np.abs(normalized_array).max()

        expected_peak = 0.8 * 32767
        assert abs(peak - expected_peak) < 1000, \
            f"Expected peak ~{expected_peak}, got {peak}"

    def test_normalize_silent_audio(self, processor):
        """Test normalizing silent audio (should return unchanged)"""
        audio = np.zeros(1600, dtype=np.int16).tobytes()

        normalized = processor.normalize_volume(audio, 8000)

        assert audio == normalized, "Silent audio should remain unchanged"


class TestDuration:
    """Test duration calculation"""

    def test_get_duration_8khz(self, processor):
        """Test duration calculation at 8kHz"""
        # 1 second = 8000 samples * 2 bytes = 16000 bytes
        audio = np.zeros(8000, dtype=np.int16).tobytes()

        duration_ms = processor.get_duration_ms(audio, 8000)

        assert abs(duration_ms - 1000) < 1, \
            f"Expected 1000ms, got {duration_ms}ms"

    def test_get_duration_16khz(self, processor):
        """Test duration calculation at 16kHz"""
        # 1 second = 16000 samples * 2 bytes = 32000 bytes
        audio = np.zeros(16000, dtype=np.int16).tobytes()

        duration_ms = processor.get_duration_ms(audio, 16000)

        assert abs(duration_ms - 1000) < 1, \
            f"Expected 1000ms, got {duration_ms}ms"
