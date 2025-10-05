"""
Integration Tests

End-to-end tests with mock voice backend.
Tests the complete call flow without real FreeSWITCH.
"""
import pytest
import pytest_asyncio
import asyncio
from sofia_phone.core.phone_handler import PhoneHandler
from sofia_phone.core.config import SofiaPhoneConfig
from tests.mocks.mock_voice_backend import MockVoiceBackend


@pytest_asyncio.fixture
async def phone_handler():
    """Phone handler with mock backend"""
    backend = MockVoiceBackend()
    handler = PhoneHandler(
        voice_backend=backend,
        esl_port=18084,  # Use different port for testing
        max_concurrent_calls=5
    )

    await handler.start()
    yield handler
    await handler.stop()


class TestPhoneHandler:
    """Test phone handler orchestration"""

    @pytest.mark.asyncio
    async def test_start_stop(self, phone_handler):
        """Test starting and stopping phone handler"""
        assert phone_handler._running
        assert phone_handler.session_manager._running

        # Metrics should be available
        metrics = phone_handler.get_metrics()
        assert "active_calls" in metrics
        assert metrics["active_calls"] == 0

    @pytest.mark.asyncio
    async def test_session_manager_integration(self, phone_handler):
        """Test session manager integration"""
        # Create a session manually
        session = phone_handler.session_manager.create_session(
            caller_id="+1234567890"
        )

        assert session is not None

        # Get metrics
        metrics = phone_handler.get_metrics()
        assert metrics["session_manager"]["active_sessions"] == 1

        # End session
        await phone_handler.session_manager.end_session(session.session_id)

        metrics = phone_handler.get_metrics()
        assert metrics["session_manager"]["active_sessions"] == 0


class TestConfig:
    """Test configuration management"""

    def test_default_config(self):
        """Test default configuration"""
        config = SofiaPhoneConfig()

        assert config.freeswitch.esl_port == 8084
        assert config.session.max_concurrent_sessions == 100
        assert config.logging.level == "INFO"

    def test_config_validation(self):
        """Test configuration validation"""
        config = SofiaPhoneConfig()

        # Should not raise
        config.validate()

    def test_config_invalid_port(self):
        """Test invalid port raises error"""
        config = SofiaPhoneConfig()
        config.freeswitch.esl_port = 99999  # Invalid

        with pytest.raises(AssertionError, match="Invalid ESL port"):
            config.validate()

    def test_config_from_dict(self):
        """Test configuration serialization"""
        config = SofiaPhoneConfig()
        config_dict = config.to_dict()

        assert config_dict["environment"] == "development"
        assert config_dict["freeswitch"]["esl_port"] == 8084


class TestMockBackend:
    """Test mock voice backend"""

    @pytest.mark.asyncio
    async def test_mock_transcribe(self):
        """Test mock transcription"""
        backend = MockVoiceBackend()

        # Generate some test audio
        audio = b"\x00" * 1000

        transcription = await backend.transcribe(audio, sample_rate=8000)

        assert "Mock transcription" in transcription
        assert "audio" in transcription

    @pytest.mark.asyncio
    async def test_mock_generate(self):
        """Test mock LLM generation"""
        backend = MockVoiceBackend()

        response = await backend.generate(
            "Hello",
            session_id="test-123",
            history=[]
        )

        assert "You said: Hello" in response

    @pytest.mark.asyncio
    async def test_mock_speak(self):
        """Test mock TTS"""
        backend = MockVoiceBackend()

        audio, sample_rate = await backend.speak("Hello world")

        assert len(audio) > 0
        assert sample_rate == 16000


class TestEndToEndFlow:
    """Test complete call flow"""

    @pytest.mark.asyncio
    async def test_conversation_flow(self):
        """Test full conversation flow with mock backend"""
        backend = MockVoiceBackend()

        # Simulate incoming audio
        audio = b"\x00" * 1000

        # 1. Transcribe
        transcription = await backend.transcribe(audio, sample_rate=8000)
        assert transcription is not None

        # 2. Generate response
        response = await backend.generate(
            transcription,
            session_id="test-123",
            history=[]
        )
        assert response is not None

        # 3. Convert to speech
        tts_audio, sample_rate = await backend.speak(response)
        assert len(tts_audio) > 0
        assert sample_rate == 16000


@pytest.mark.asyncio
async def test_concurrent_sessions():
    """Test handling multiple concurrent sessions"""
    backend = MockVoiceBackend()
    handler = PhoneHandler(
        voice_backend=backend,
        esl_port=18085,
        max_concurrent_calls=10
    )

    await handler.start()

    try:
        # Create multiple sessions
        sessions = []
        for i in range(5):
            session = handler.session_manager.create_session(
                caller_id=f"+{i:010d}"
            )
            sessions.append(session)

        # All sessions should be isolated
        assert len(set(s.session_id for s in sessions)) == 5

        # Get metrics
        metrics = handler.get_metrics()
        assert metrics["session_manager"]["active_sessions"] == 5

    finally:
        await handler.stop()
