"""
Production-Grade Session Manager

Manages multiple concurrent call sessions with complete isolation.
Each caller gets their own conversation history and state.

Features:
- Thread-safe session management
- Conversation history per caller
- Session lifecycle tracking
- Resource cleanup
- Metrics and monitoring
- Automatic session expiry
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import uuid
from loguru import logger
import threading


class SessionState(Enum):
    """Call session states"""
    CREATED = "created"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    TRANSFERRING = "transferring"
    ENDING = "ending"
    ENDED = "ended"
    ERROR = "error"


@dataclass
class CallSession:
    """
    Represents a single call session.

    Each session is completely isolated - separate conversation history,
    separate AI backend instance, separate state.
    """
    session_id: str
    caller_id: str
    created_at: datetime = field(default_factory=datetime.now)
    state: SessionState = SessionState.CREATED
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_activity: datetime = field(default_factory=datetime.now)

    # Metrics
    audio_chunks_received: int = 0
    audio_chunks_sent: int = 0
    transcriptions_count: int = 0
    llm_responses_count: int = 0
    errors_count: int = 0

    def update_activity(self):
        """Mark session as active (for timeout tracking)"""
        self.last_activity = datetime.now()

    def add_conversation_turn(self, user_message: str, ai_response: str):
        """
        Add conversation turn to history.

        Args:
            user_message: What user said
            ai_response: AI's response
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.now().isoformat()
        })
        self.update_activity()

    def get_duration(self) -> timedelta:
        """Get call duration"""
        return datetime.now() - self.created_at

    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session expired due to inactivity"""
        inactive_time = datetime.now() - self.last_activity
        return inactive_time > timedelta(minutes=timeout_minutes)

    def __str__(self) -> str:
        duration = self.get_duration().total_seconds()
        return (
            f"CallSession({self.session_id[:8]}...{self.caller_id}, "
            f"state={self.state.value}, duration={duration:.1f}s, "
            f"turns={len(self.conversation_history)//2})"
        )


class SessionManager:
    """
    Production-grade multi-session manager.

    Thread-safe management of concurrent call sessions.
    Handles session lifecycle, cleanup, and monitoring.
    """

    def __init__(
        self,
        max_sessions: int = 100,
        session_timeout_minutes: int = 30,
        cleanup_interval_seconds: int = 60
    ):
        """
        Initialize session manager.

        Args:
            max_sessions: Maximum concurrent sessions
            session_timeout_minutes: Auto-cleanup inactive sessions
            cleanup_interval_seconds: How often to run cleanup
        """
        self._sessions: Dict[str, CallSession] = {}
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self.max_sessions = max_sessions
        self.session_timeout_minutes = session_timeout_minutes
        self.cleanup_interval_seconds = cleanup_interval_seconds

        # Metrics
        self.total_sessions_created = 0
        self.total_sessions_ended = 0

        # Start background cleanup task
        self._cleanup_task = None
        self._running = False

        logger.info(
            f"SessionManager initialized (max={max_sessions}, "
            f"timeout={session_timeout_minutes}min)"
        )

    async def start(self):
        """Start background cleanup task"""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("SessionManager background cleanup started")

    async def stop(self):
        """Stop background cleanup and close all sessions"""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all active sessions
        with self._lock:
            for session in list(self._sessions.values()):
                await self.end_session(session.session_id)

        logger.info("SessionManager stopped")

    async def _cleanup_loop(self):
        """Background task to cleanup expired sessions"""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def _cleanup_expired_sessions(self):
        """Remove sessions that have been inactive too long"""
        with self._lock:
            expired = [
                session_id
                for session_id, session in self._sessions.items()
                if session.is_expired(self.session_timeout_minutes)
            ]

        if expired:
            logger.info(f"Cleaning up {len(expired)} expired sessions")
            for session_id in expired:
                await self.end_session(session_id, reason="timeout")

    def create_session(
        self,
        caller_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CallSession:
        """
        Create a new call session.

        Args:
            caller_id: Caller's phone number
            metadata: Optional metadata (caller name, etc.)

        Returns:
            New CallSession

        Raises:
            RuntimeError: If max sessions reached
        """
        with self._lock:
            # Check capacity
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError(
                    f"Maximum sessions reached ({self.max_sessions}). "
                    "Cannot create new session."
                )

            # Generate unique session ID
            session_id = f"call-{uuid.uuid4().hex[:16]}"

            # Create session
            session = CallSession(
                session_id=session_id,
                caller_id=caller_id,
                metadata=metadata or {}
            )

            self._sessions[session_id] = session
            self.total_sessions_created += 1

            logger.info(f"Session created: {session}")
            return session

    def get_session(self, session_id: str) -> Optional[CallSession]:
        """
        Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            CallSession or None if not found
        """
        with self._lock:
            return self._sessions.get(session_id)

    def update_session_state(
        self,
        session_id: str,
        state: SessionState
    ) -> Optional[CallSession]:
        """
        Update session state.

        Args:
            session_id: Session identifier
            state: New state

        Returns:
            Updated session or None if not found
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                old_state = session.state
                session.state = state
                session.update_activity()
                logger.debug(f"Session {session_id[:8]} state: {old_state.value} → {state.value}")
            return session

    async def end_session(
        self,
        session_id: str,
        reason: str = "normal"
    ) -> Optional[CallSession]:
        """
        End a call session and cleanup resources.

        Args:
            session_id: Session identifier
            reason: Reason for ending (normal, timeout, error, etc.)

        Returns:
            Ended session or None if not found
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                logger.warning(f"Cannot end session {session_id}: not found")
                return None

            # Mark as ended
            session.state = SessionState.ENDED
            duration = session.get_duration().total_seconds()

            # Log metrics
            logger.info(
                f"Session ended: {session.session_id[:8]} (reason={reason}, "
                f"duration={duration:.1f}s, turns={len(session.conversation_history)//2}, "
                f"errors={session.errors_count})"
            )

            # Remove from active sessions
            del self._sessions[session_id]
            self.total_sessions_ended += 1

            return session

    def get_active_sessions(self) -> List[CallSession]:
        """Get all active sessions"""
        with self._lock:
            return [
                session for session in self._sessions.values()
                if session.state == SessionState.ACTIVE
            ]

    def get_session_count(self) -> int:
        """Get count of active sessions"""
        with self._lock:
            return len(self._sessions)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get session manager metrics.

        Returns:
            Dictionary with metrics
        """
        with self._lock:
            active_count = len(self._sessions)
            active_sessions = list(self._sessions.values())

            return {
                "active_sessions": active_count,
                "total_created": self.total_sessions_created,
                "total_ended": self.total_sessions_ended,
                "max_sessions": self.max_sessions,
                "sessions": [
                    {
                        "session_id": s.session_id,
                        "caller_id": s.caller_id,
                        "state": s.state.value,
                        "duration_seconds": s.get_duration().total_seconds(),
                        "conversation_turns": len(s.conversation_history) // 2,
                        "errors": s.errors_count
                    }
                    for s in active_sessions
                ]
            }

    def __str__(self) -> str:
        with self._lock:
            return (
                f"SessionManager(active={len(self._sessions)}/{self.max_sessions}, "
                f"created={self.total_sessions_created}, ended={self.total_sessions_ended})"
            )
