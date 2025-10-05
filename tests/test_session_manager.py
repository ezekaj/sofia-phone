"""
Tests for SessionManager

Tests session management functionality:
- Session creation and lifecycle
- Multi-session isolation
- Session expiry
- Thread safety
- Metrics
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from sofia_phone.core.session_manager import (
    SessionManager,
    SessionState,
    CallSession
)


@pytest.fixture
async def manager():
    """Session manager fixture"""
    manager = SessionManager(max_sessions=10)
    await manager.start()
    yield manager
    await manager.stop()


class TestSessionCreation:
    """Test session creation"""

    @pytest.mark.asyncio
    async def test_create_session(self, manager):
        """Test creating a session"""
        session = manager.create_session(
            caller_id="+1234567890",
            metadata={"test": "data"}
        )

        assert session is not None
        assert session.caller_id == "+1234567890"
        assert session.metadata["test"] == "data"
        assert session.state == SessionState.CREATED
        assert len(session.conversation_history) == 0

    @pytest.mark.asyncio
    async def test_session_id_unique(self, manager):
        """Test that session IDs are unique"""
        session1 = manager.create_session(caller_id="+1111111111")
        session2 = manager.create_session(caller_id="+1111111111")

        assert session1.session_id != session2.session_id

    @pytest.mark.asyncio
    async def test_max_sessions_enforced(self, manager):
        """Test that max sessions limit is enforced"""
        # Create max sessions
        for i in range(10):
            manager.create_session(caller_id=f"+{i}")

        # Try to create one more (should fail)
        with pytest.raises(RuntimeError, match="Maximum sessions reached"):
            manager.create_session(caller_id="+9999999999")


class TestSessionRetrieval:
    """Test session retrieval"""

    @pytest.mark.asyncio
    async def test_get_session(self, manager):
        """Test retrieving a session"""
        session = manager.create_session(caller_id="+1234567890")

        retrieved = manager.get_session(session.session_id)

        assert retrieved is not None
        assert retrieved.session_id == session.session_id
        assert retrieved.caller_id == "+1234567890"

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, manager):
        """Test retrieving non-existent session"""
        retrieved = manager.get_session("nonexistent-id")

        assert retrieved is None


class TestSessionState:
    """Test session state management"""

    @pytest.mark.asyncio
    async def test_update_state(self, manager):
        """Test updating session state"""
        session = manager.create_session(caller_id="+1234567890")

        updated = manager.update_session_state(
            session.session_id,
            SessionState.ACTIVE
        )

        assert updated is not None
        assert updated.state == SessionState.ACTIVE

    @pytest.mark.asyncio
    async def test_state_transition(self, manager):
        """Test full state lifecycle"""
        session = manager.create_session(caller_id="+1234567890")

        # CREATED → ACTIVE
        manager.update_session_state(session.session_id, SessionState.ACTIVE)
        assert manager.get_session(session.session_id).state == SessionState.ACTIVE

        # ACTIVE → ON_HOLD
        manager.update_session_state(session.session_id, SessionState.ON_HOLD)
        assert manager.get_session(session.session_id).state == SessionState.ON_HOLD

        # ON_HOLD → ACTIVE
        manager.update_session_state(session.session_id, SessionState.ACTIVE)
        assert manager.get_session(session.session_id).state == SessionState.ACTIVE


class TestConversationHistory:
    """Test conversation history tracking"""

    @pytest.mark.asyncio
    async def test_add_conversation_turn(self, manager):
        """Test adding conversation turn"""
        session = manager.create_session(caller_id="+1234567890")

        session.add_conversation_turn(
            user_message="Hello",
            ai_response="Hi there!"
        )

        assert len(session.conversation_history) == 2
        assert session.conversation_history[0]["role"] == "user"
        assert session.conversation_history[0]["content"] == "Hello"
        assert session.conversation_history[1]["role"] == "assistant"
        assert session.conversation_history[1]["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_multiple_turns(self, manager):
        """Test multiple conversation turns"""
        session = manager.create_session(caller_id="+1234567890")

        session.add_conversation_turn("Message 1", "Response 1")
        session.add_conversation_turn("Message 2", "Response 2")
        session.add_conversation_turn("Message 3", "Response 3")

        # Should have 6 entries (3 user + 3 assistant)
        assert len(session.conversation_history) == 6


class TestSessionLifecycle:
    """Test session lifecycle management"""

    @pytest.mark.asyncio
    async def test_end_session(self, manager):
        """Test ending a session"""
        session = manager.create_session(caller_id="+1234567890")
        session_id = session.session_id

        # End session
        ended = await manager.end_session(session_id, reason="test")

        assert ended is not None
        assert ended.state == SessionState.ENDED

        # Session should be removed from active sessions
        assert manager.get_session(session_id) is None

    @pytest.mark.asyncio
    async def test_session_duration(self, manager):
        """Test session duration calculation"""
        session = manager.create_session(caller_id="+1234567890")

        # Wait a bit
        await asyncio.sleep(0.1)

        duration = session.get_duration()

        assert duration.total_seconds() >= 0.1

    @pytest.mark.asyncio
    async def test_session_expiry(self, manager):
        """Test session expiry detection"""
        session = manager.create_session(caller_id="+1234567890")

        # Manually set last_activity to 31 minutes ago
        session.last_activity = datetime.now() - timedelta(minutes=31)

        # Should be expired (timeout is 30 minutes)
        assert session.is_expired(timeout_minutes=30)

    @pytest.mark.asyncio
    async def test_auto_cleanup_expired(self, manager):
        """Test automatic cleanup of expired sessions"""
        # Create session
        session = manager.create_session(caller_id="+1234567890")

        # Force expiry
        session.last_activity = datetime.now() - timedelta(minutes=31)

        # Trigger cleanup
        await manager._cleanup_expired_sessions()

        # Session should be removed
        assert manager.get_session(session.session_id) is None


class TestMultipleSessions:
    """Test concurrent session management"""

    @pytest.mark.asyncio
    async def test_multiple_callers(self, manager):
        """Test handling multiple callers simultaneously"""
        session1 = manager.create_session(caller_id="+1111111111")
        session2 = manager.create_session(caller_id="+2222222222")
        session3 = manager.create_session(caller_id="+3333333333")

        # Each should have separate state
        manager.update_session_state(session1.session_id, SessionState.ACTIVE)
        manager.update_session_state(session2.session_id, SessionState.ON_HOLD)

        assert manager.get_session(session1.session_id).state == SessionState.ACTIVE
        assert manager.get_session(session2.session_id).state == SessionState.ON_HOLD
        assert manager.get_session(session3.session_id).state == SessionState.CREATED

    @pytest.mark.asyncio
    async def test_isolated_conversation_history(self, manager):
        """Test that conversation histories are isolated"""
        session1 = manager.create_session(caller_id="+1111111111")
        session2 = manager.create_session(caller_id="+2222222222")

        # Add conversations to each
        session1.add_conversation_turn("Hello from caller 1", "Response 1")
        session2.add_conversation_turn("Hello from caller 2", "Response 2")

        # Histories should be separate
        assert len(session1.conversation_history) == 2
        assert len(session2.conversation_history) == 2
        assert session1.conversation_history[0]["content"] == "Hello from caller 1"
        assert session2.conversation_history[0]["content"] == "Hello from caller 2"

    @pytest.mark.asyncio
    async def test_get_active_sessions(self, manager):
        """Test getting only active sessions"""
        session1 = manager.create_session(caller_id="+1111111111")
        session2 = manager.create_session(caller_id="+2222222222")
        session3 = manager.create_session(caller_id="+3333333333")

        manager.update_session_state(session1.session_id, SessionState.ACTIVE)
        manager.update_session_state(session2.session_id, SessionState.ACTIVE)
        manager.update_session_state(session3.session_id, SessionState.CREATED)

        active = manager.get_active_sessions()

        assert len(active) == 2
        assert all(s.state == SessionState.ACTIVE for s in active)


class TestMetrics:
    """Test metrics tracking"""

    @pytest.mark.asyncio
    async def test_session_count(self, manager):
        """Test session count"""
        assert manager.get_session_count() == 0

        manager.create_session(caller_id="+1111111111")
        assert manager.get_session_count() == 1

        manager.create_session(caller_id="+2222222222")
        assert manager.get_session_count() == 2

    @pytest.mark.asyncio
    async def test_metrics(self, manager):
        """Test metrics collection"""
        session1 = manager.create_session(caller_id="+1111111111")
        session2 = manager.create_session(caller_id="+2222222222")

        metrics = manager.get_metrics()

        assert metrics["active_sessions"] == 2
        assert metrics["total_created"] == 2
        assert metrics["total_ended"] == 0
        assert metrics["max_sessions"] == 10
        assert len(metrics["sessions"]) == 2

    @pytest.mark.asyncio
    async def test_session_metrics(self, manager):
        """Test individual session metrics"""
        session = manager.create_session(caller_id="+1234567890")

        # Simulate activity
        session.audio_chunks_received = 100
        session.transcriptions_count = 5
        session.llm_responses_count = 5
        session.errors_count = 2

        metrics = manager.get_metrics()
        session_metrics = metrics["sessions"][0]

        assert session_metrics["errors"] == 2
        assert session_metrics["conversation_turns"] == 0  # No conversations added yet
