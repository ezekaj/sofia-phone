"""
Production-Grade Configuration Management

Centralized configuration with:
- Environment variables (12-factor app)
- Configuration validation
- Type safety
- Secure defaults
- Easy testing (override configs)

All settings in one place for easy management.
"""
import os
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


@dataclass
class FreeSWITCHConfig:
    """FreeSWITCH configuration"""
    # ESL settings
    esl_host: str = "0.0.0.0"
    esl_port: int = 8084
    esl_password: Optional[str] = None  # Not needed for outbound mode

    # SIP settings
    sip_host: str = "127.0.0.1"
    sip_port: int = 5060

    # RTP settings
    rtp_start_port: int = 16384
    rtp_end_port: int = 32768

    # Audio settings
    sample_rate: int = 8000  # G.711 standard
    frame_duration_ms: int = 20  # Standard ptime

    def validate(self):
        """Validate configuration"""
        assert 1 <= self.esl_port <= 65535, "Invalid ESL port"
        assert 1 <= self.sip_port <= 65535, "Invalid SIP port"
        assert self.rtp_start_port < self.rtp_end_port, "Invalid RTP port range"
        assert self.sample_rate in [8000, 16000, 48000], "Unsupported sample rate"


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    # Sample rates
    telephony_sample_rate: int = 8000  # FreeSWITCH
    whisper_sample_rate: int = 16000  # Whisper STT
    tts_sample_rate: int = 24000  # Edge TTS default

    # Quality thresholds
    silence_threshold: int = 500  # Amplitude threshold
    clipping_threshold: int = 30000  # Near max int16
    min_audio_duration_ms: int = 100  # Minimum viable audio

    # Voice activity detection
    vad_enabled: bool = True
    vad_aggressiveness: int = 2  # 0-3 (0=lenient, 3=aggressive)

    # Volume normalization
    normalize_volume: bool = True
    target_peak_level: float = 0.8  # 80% of max

    def validate(self):
        """Validate configuration"""
        assert self.telephony_sample_rate > 0, "Invalid telephony sample rate"
        assert self.whisper_sample_rate > 0, "Invalid whisper sample rate"
        assert self.tts_sample_rate > 0, "Invalid TTS sample rate"
        assert 0 <= self.vad_aggressiveness <= 3, "Invalid VAD aggressiveness"
        assert 0.0 < self.target_peak_level <= 1.0, "Invalid target peak level"


@dataclass
class SessionConfig:
    """Session management configuration"""
    # Capacity
    max_concurrent_sessions: int = 100
    max_sessions_per_caller: int = 1  # Prevent spam

    # Timeouts
    session_timeout_minutes: int = 30  # Auto-cleanup inactive
    cleanup_interval_seconds: int = 60  # Cleanup task frequency

    # Conversation
    max_conversation_history: int = 50  # Limit history size
    conversation_context_turns: int = 10  # How many turns to send to LLM

    def validate(self):
        """Validate configuration"""
        assert self.max_concurrent_sessions > 0, "Invalid max sessions"
        assert self.session_timeout_minutes > 0, "Invalid session timeout"
        assert self.max_conversation_history >= self.conversation_context_turns


@dataclass
class ConversationConfig:
    """Conversation flow configuration"""
    # Transcription settings
    min_transcription_duration_ms: int = 500  # Min audio for STT
    max_silence_before_transcription_ms: int = 800  # Silence threshold

    # Interruption detection
    interruption_enabled: bool = True
    interruption_threshold_chunks: int = 5  # Require sustained speech (100ms)

    # Response generation
    llm_timeout_seconds: float = 30.0
    llm_max_tokens: int = 200  # Keep responses concise

    # TTS settings
    tts_timeout_seconds: float = 10.0
    tts_chunk_size_bytes: int = 320  # 20ms at 8kHz (160 samples * 2 bytes)

    def validate(self):
        """Validate configuration"""
        assert self.min_transcription_duration_ms > 0
        assert self.max_silence_before_transcription_ms > 0
        assert self.interruption_threshold_chunks > 0
        assert self.llm_timeout_seconds > 0
        assert self.tts_timeout_seconds > 0


@dataclass
class ErrorRecoveryConfig:
    """Error recovery configuration"""
    # Circuit breaker
    circuit_failure_threshold: int = 5
    circuit_success_threshold: int = 2
    circuit_timeout_seconds: float = 60.0

    # Retry policy
    max_retry_attempts: int = 3
    initial_retry_delay_seconds: float = 1.0
    max_retry_delay_seconds: float = 60.0
    retry_exponential_base: float = 2.0
    retry_jitter_enabled: bool = True

    def validate(self):
        """Validate configuration"""
        assert self.circuit_failure_threshold > 0
        assert self.circuit_success_threshold > 0
        assert self.circuit_timeout_seconds > 0
        assert self.max_retry_attempts > 0
        assert self.initial_retry_delay_seconds > 0


@dataclass
class LoggingConfig:
    """Logging configuration"""
    # Log level
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # Output
    console_enabled: bool = True
    file_enabled: bool = True
    file_path: str = "logs/sofia-phone.log"
    file_rotation: str = "100 MB"  # Rotate at size
    file_retention: str = "1 week"  # Keep for duration

    # Format
    json_logs: bool = False  # JSON for production
    colorize: bool = True  # Color in console

    # Filters
    log_audio_metrics: bool = True
    log_conversation: bool = True
    log_rtp_packets: bool = False  # Very verbose

    def validate(self):
        """Validate configuration"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert self.level in valid_levels, f"Invalid log level (must be one of {valid_levels})"


@dataclass
class HealthCheckConfig:
    """Health check configuration"""
    # Endpoints
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080

    # Checks
    check_interval_seconds: int = 30
    unhealthy_threshold: int = 3  # Consecutive failures

    # Metrics
    metrics_enabled: bool = True
    metrics_retention_seconds: int = 3600  # 1 hour

    def validate(self):
        """Validate configuration"""
        assert 1 <= self.port <= 65535, "Invalid health check port"
        assert self.check_interval_seconds > 0


@dataclass
class SofiaPhoneConfig:
    """
    Main configuration for sofia-phone.

    All settings centralized here for easy management.
    Override via environment variables (12-factor app).
    """
    # Component configs
    freeswitch: FreeSWITCHConfig = field(default_factory=FreeSWITCHConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    error_recovery: ErrorRecoveryConfig = field(default_factory=ErrorRecoveryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)

    # General settings
    environment: str = "development"  # development, production
    debug: bool = False

    @classmethod
    def from_env(cls) -> "SofiaPhoneConfig":
        """
        Load configuration from environment variables.

        Environment variables override defaults:
        - SOFIA_PHONE_ENV: development/production
        - SOFIA_PHONE_DEBUG: true/false
        - SOFIA_PHONE_ESL_PORT: ESL port
        - SOFIA_PHONE_MAX_SESSIONS: Max concurrent sessions
        - etc.

        Returns:
            Configuration instance
        """
        config = cls()

        # General
        config.environment = os.getenv("SOFIA_PHONE_ENV", "development")
        config.debug = os.getenv("SOFIA_PHONE_DEBUG", "false").lower() == "true"

        # FreeSWITCH
        config.freeswitch.esl_host = os.getenv(
            "SOFIA_PHONE_ESL_HOST",
            config.freeswitch.esl_host
        )
        config.freeswitch.esl_port = int(os.getenv(
            "SOFIA_PHONE_ESL_PORT",
            str(config.freeswitch.esl_port)
        ))
        config.freeswitch.rtp_start_port = int(os.getenv(
            "SOFIA_PHONE_RTP_START_PORT",
            str(config.freeswitch.rtp_start_port)
        ))

        # Session
        config.session.max_concurrent_sessions = int(os.getenv(
            "SOFIA_PHONE_MAX_SESSIONS",
            str(config.session.max_concurrent_sessions)
        ))
        config.session.session_timeout_minutes = int(os.getenv(
            "SOFIA_PHONE_SESSION_TIMEOUT",
            str(config.session.session_timeout_minutes)
        ))

        # Logging
        config.logging.level = os.getenv(
            "SOFIA_PHONE_LOG_LEVEL",
            config.logging.level
        ).upper()
        config.logging.file_path = os.getenv(
            "SOFIA_PHONE_LOG_FILE",
            config.logging.file_path
        )
        config.logging.json_logs = os.getenv(
            "SOFIA_PHONE_JSON_LOGS",
            "false"
        ).lower() == "true"

        # Health check
        config.health_check.enabled = os.getenv(
            "SOFIA_PHONE_HEALTH_CHECK",
            "true"
        ).lower() == "true"
        config.health_check.port = int(os.getenv(
            "SOFIA_PHONE_HEALTH_PORT",
            str(config.health_check.port)
        ))

        # Production defaults
        if config.environment == "production":
            config.debug = False
            config.logging.json_logs = True
            config.logging.colorize = False
            config.logging.log_rtp_packets = False

        return config

    def validate(self):
        """Validate all configuration"""
        self.freeswitch.validate()
        self.audio.validate()
        self.session.validate()
        self.conversation.validate()
        self.error_recovery.validate()
        self.logging.validate()
        self.health_check.validate()

        logger.info("Configuration validated successfully")

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/debugging"""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "freeswitch": {
                "esl_host": self.freeswitch.esl_host,
                "esl_port": self.freeswitch.esl_port,
                "rtp_port_range": f"{self.freeswitch.rtp_start_port}-{self.freeswitch.rtp_end_port}",
            },
            "session": {
                "max_concurrent": self.session.max_concurrent_sessions,
                "timeout_minutes": self.session.session_timeout_minutes,
            },
            "logging": {
                "level": self.logging.level,
                "file": self.logging.file_path if self.logging.file_enabled else None,
                "json": self.logging.json_logs,
            },
            "health_check": {
                "enabled": self.health_check.enabled,
                "port": self.health_check.port if self.health_check.enabled else None,
            }
        }

    def __str__(self) -> str:
        return (
            f"SofiaPhoneConfig("
            f"env={self.environment}, "
            f"esl={self.freeswitch.esl_port}, "
            f"max_sessions={self.session.max_concurrent_sessions}, "
            f"log_level={self.logging.level}"
            f")"
        )


# Global config instance
_config: Optional[SofiaPhoneConfig] = None


def get_config() -> SofiaPhoneConfig:
    """
    Get global configuration instance.

    Loads from environment on first call.

    Returns:
        Configuration instance
    """
    global _config

    if _config is None:
        _config = SofiaPhoneConfig.from_env()
        _config.validate()
        logger.info(f"Configuration loaded: {_config}")

    return _config


def set_config(config: SofiaPhoneConfig):
    """
    Set global configuration (for testing).

    Args:
        config: Configuration instance
    """
    global _config
    _config = config
    _config.validate()
