"""
Production-Grade Logging System

Features:
- Structured logging with loguru
- Console + file output
- Automatic rotation
- JSON format for production
- Context-aware logging (session IDs, caller IDs)
- Performance metrics
- Error tracking
- Log levels per module

Best Practices:
- All logs include timestamps, module, function
- Sensitive data (phone numbers) partially masked in production
- JSON logs for machine parsing (ELK, Splunk, etc.)
- Contextual information (session ID, call ID)
"""
import sys
import os
from pathlib import Path
from loguru import logger
from typing import Optional
import json
from datetime import datetime

from .config import LoggingConfig


class StructuredLogger:
    """
    Structured logging with context.

    Wraps loguru to add context (session_id, caller_id, etc.)
    to all log messages.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        caller_id: Optional[str] = None,
        call_id: Optional[str] = None
    ):
        """
        Initialize structured logger.

        Args:
            session_id: Session identifier
            caller_id: Caller phone number
            call_id: FreeSWITCH call ID
        """
        self.context = {}

        if session_id:
            self.context["session_id"] = session_id[:8]  # First 8 chars only

        if caller_id:
            # Mask phone number in production (e.g., +1234***890)
            self.context["caller_id"] = self._mask_phone(caller_id)

        if call_id:
            self.context["call_id"] = call_id[:8]

    def _mask_phone(self, phone: str) -> str:
        """Mask phone number for privacy (keep first 4 and last 3 digits)"""
        if len(phone) <= 7:
            return phone

        return f"{phone[:4]}***{phone[-3:]}"

    def _log(self, level: str, message: str, **kwargs):
        """Internal log method with context"""
        # Merge context into kwargs
        log_data = {**self.context, **kwargs}

        # Format message with context
        if log_data:
            context_str = " | ".join([f"{k}={v}" for k, v in log_data.items()])
            message = f"{message} | {context_str}"

        # Log at appropriate level
        getattr(logger, level)(message)

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log("debug", message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log("info", message, **kwargs)

    def success(self, message: str, **kwargs):
        """Log success message"""
        self._log("success", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log("warning", message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log("error", message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log("critical", message, **kwargs)


def setup_logging(config: LoggingConfig):
    """
    Setup production logging system.

    Configures loguru with:
    - Console output (with colors in dev)
    - File output (with rotation)
    - JSON format for production
    - Appropriate log levels

    Args:
        config: Logging configuration
    """
    # Remove default handler
    logger.remove()

    # Console handler
    if config.console_enabled:
        if config.json_logs:
            # JSON format for production
            logger.add(
                sys.stdout,
                format=_json_formatter,
                level=config.level,
                colorize=False,
                serialize=False  # We handle JSON ourselves
            )
        else:
            # Pretty format for development
            logger.add(
                sys.stdout,
                format=(
                    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                    "<level>{level: <8}</level> | "
                    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                    "<level>{message}</level>"
                ),
                level=config.level,
                colorize=config.colorize
            )

    # File handler
    if config.file_enabled:
        # Create log directory
        log_path = Path(config.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if config.json_logs:
            # JSON format for production
            logger.add(
                config.file_path,
                format=_json_formatter,
                level=config.level,
                rotation=config.file_rotation,
                retention=config.file_retention,
                compression="zip",
                serialize=False
            )
        else:
            # Pretty format for development
            logger.add(
                config.file_path,
                format=(
                    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                    "{name}:{function}:{line} | {message}"
                ),
                level=config.level,
                rotation=config.file_rotation,
                retention=config.file_retention,
                compression="zip"
            )

    logger.info(
        f"Logging configured: level={config.level}, "
        f"console={config.console_enabled}, file={config.file_enabled}, "
        f"json={config.json_logs}"
    )


def _json_formatter(record: dict) -> str:
    """
    Format log record as JSON.

    Args:
        record: Log record

    Returns:
        JSON string
    """
    # Extract relevant fields
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
    }

    # Add exception if present
    if record["exception"]:
        log_entry["exception"] = {
            "type": record["exception"].type.__name__,
            "value": str(record["exception"].value),
            "traceback": record["exception"].traceback
        }

    # Add extra fields if present
    if record["extra"]:
        log_entry["extra"] = record["extra"]

    return json.dumps(log_entry) + "\n"


class PerformanceLogger:
    """
    Logger for performance metrics.

    Tracks timing, counts, and performance data.
    """

    def __init__(self, name: str):
        """
        Initialize performance logger.

        Args:
            name: Metric name
        """
        self.name = name
        self.start_time = None

    def __enter__(self):
        """Start timing"""
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log"""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds() * 1000
            logger.debug(f"Performance: {self.name} took {duration:.2f}ms")


def log_performance(name: str):
    """
    Decorator for logging function performance.

    Usage:
        @log_performance("my_function")
        async def my_function():
            ...

    Args:
        name: Metric name
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = await func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds() * 1000
                logger.debug(f"Performance: {name} took {duration:.2f}ms")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds() * 1000
                logger.error(f"Performance: {name} failed after {duration:.2f}ms: {e}")
                raise
        return wrapper
    return decorator


class MetricsCollector:
    """
    Collect and log metrics.

    Tracks counts, rates, and distributions.
    """

    def __init__(self):
        """Initialize metrics collector"""
        self.metrics = {}

    def increment(self, metric: str, value: int = 1):
        """
        Increment counter.

        Args:
            metric: Metric name
            value: Increment value
        """
        if metric not in self.metrics:
            self.metrics[metric] = 0

        self.metrics[metric] += value

    def set(self, metric: str, value: float):
        """
        Set gauge value.

        Args:
            metric: Metric name
            value: Metric value
        """
        self.metrics[metric] = value

    def log_metrics(self):
        """Log all metrics"""
        if self.metrics:
            logger.info(f"Metrics: {json.dumps(self.metrics)}")

    def get_metrics(self) -> dict:
        """Get all metrics"""
        return self.metrics.copy()

    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()


# Global metrics collector
_metrics = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector"""
    return _metrics


def log_call_event(
    event: str,
    session_id: str,
    caller_id: str,
    **kwargs
):
    """
    Log call event with standard format.

    Args:
        event: Event name (call_start, call_end, etc.)
        session_id: Session identifier
        caller_id: Caller phone number
        **kwargs: Additional event data
    """
    structured_logger = StructuredLogger(
        session_id=session_id,
        caller_id=caller_id
    )

    structured_logger.info(f"Call event: {event}", **kwargs)


def log_audio_metrics(
    session_id: str,
    direction: str,  # inbound/outbound
    duration_ms: float,
    sample_rate: int,
    size_bytes: int,
    **kwargs
):
    """
    Log audio metrics.

    Args:
        session_id: Session identifier
        direction: inbound/outbound
        duration_ms: Audio duration in milliseconds
        sample_rate: Sample rate in Hz
        size_bytes: Audio size in bytes
        **kwargs: Additional metrics
    """
    structured_logger = StructuredLogger(session_id=session_id)

    structured_logger.debug(
        f"Audio {direction}",
        duration_ms=f"{duration_ms:.1f}ms",
        sample_rate=f"{sample_rate}Hz",
        size=f"{size_bytes}B",
        **kwargs
    )


def log_conversation_turn(
    session_id: str,
    caller_id: str,
    user_text: str,
    ai_text: str,
    stt_duration_ms: float,
    llm_duration_ms: float,
    tts_duration_ms: float
):
    """
    Log conversation turn with timing.

    Args:
        session_id: Session identifier
        caller_id: Caller phone number
        user_text: User transcription
        ai_text: AI response
        stt_duration_ms: STT processing time
        llm_duration_ms: LLM processing time
        tts_duration_ms: TTS processing time
    """
    structured_logger = StructuredLogger(
        session_id=session_id,
        caller_id=caller_id
    )

    total_ms = stt_duration_ms + llm_duration_ms + tts_duration_ms

    structured_logger.info(
        "Conversation turn",
        user=user_text[:50] + "..." if len(user_text) > 50 else user_text,
        ai=ai_text[:50] + "..." if len(ai_text) > 50 else ai_text,
        stt_ms=f"{stt_duration_ms:.0f}",
        llm_ms=f"{llm_duration_ms:.0f}",
        tts_ms=f"{tts_duration_ms:.0f}",
        total_ms=f"{total_ms:.0f}"
    )
