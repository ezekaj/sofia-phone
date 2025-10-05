"""
Sofia-Phone Main Entry Point

Production-ready phone system orchestrator.
Starts all components and handles graceful shutdown.
"""
import asyncio
import signal
from loguru import logger

from .core.config import get_config
from .core.logging_setup import setup_logging
from .core.phone_handler import PhoneHandler
from .core.health import HealthChecker, check_freeswich_health, check_session_manager_health
from .core.error_recovery import ErrorRecoveryManager
from .interfaces.voice_backend import VoiceBackend


async def main():
    """
    Main application entry point.

    Starts all components and handles graceful shutdown.
    """
    # Load configuration
    config = get_config()

    # Setup logging
    setup_logging(config.logging)

    logger.info("=" * 60)
    logger.info("Sofia-Phone Starting")
    logger.info("=" * 60)
    logger.info(f"Environment: {config.environment}")
    logger.info(f"ESL Port: {config.freeswitch.esl_port}")
    logger.info(f"Max Sessions: {config.session.max_concurrent_sessions}")
    logger.info(f"Health Check Port: {config.health_check.port}")
    logger.info("=" * 60)

    # Get voice backend from environment or use mock
    # In production, this would be injected
    try:
        # Try to import real backend (sofia-ultimate integration)
        from sofia_ultimate.voice_backend import SofiaVoiceBackend
        voice_backend = SofiaVoiceBackend()
        logger.info("Using SofiaVoiceBackend (production)")
    except ImportError:
        # Fall back to mock for testing
        from tests.mocks.mock_voice_backend import MockVoiceBackend
        voice_backend = MockVoiceBackend()
        logger.warning("Using MockVoiceBackend (development/testing)")

    # Initialize components
    error_recovery = ErrorRecoveryManager()

    phone_handler = PhoneHandler(
        voice_backend=voice_backend,
        esl_host=config.freeswitch.esl_host,
        esl_port=config.freeswitch.esl_port,
        rtp_start_port=config.freeswitch.rtp_start_port,
        max_concurrent_calls=config.session.max_concurrent_sessions
    )

    health_checker = None
    if config.health_check.enabled:
        health_checker = HealthChecker(
            host=config.health_check.host,
            port=config.health_check.port,
            check_interval_seconds=config.health_check.check_interval_seconds
        )

        # Register health checks
        health_checker.register_check(
            "freeswitch",
            lambda: check_freeswich_health(phone_handler)
        )
        health_checker.register_check(
            "session_manager",
            lambda: check_session_manager_health(phone_handler.session_manager)
        )

    # Graceful shutdown handler
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {sig}, initiating graceful shutdown...")
        shutdown_event.set()

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Start all components
        logger.info("Starting components...")

        await phone_handler.start()

        if health_checker:
            await health_checker.start()

        logger.success("=" * 60)
        logger.success("Sofia-Phone Running")
        logger.success("=" * 60)
        logger.success(f"ESL Server: {config.freeswitch.esl_host}:{config.freeswitch.esl_port}")
        if config.health_check.enabled:
            logger.success(f"Health Checks: http://{config.health_check.host}:{config.health_check.port}/health")
        logger.success("Ready to accept calls!")
        logger.success("=" * 60)

        # Wait for shutdown signal
        await shutdown_event.wait()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise

    finally:
        # Graceful shutdown
        logger.info("=" * 60)
        logger.info("Shutting down Sofia-Phone...")
        logger.info("=" * 60)

        if health_checker:
            logger.info("Stopping health checker...")
            await health_checker.stop()

        logger.info("Stopping phone handler...")
        await phone_handler.stop()

        logger.info("=" * 60)
        logger.info("Sofia-Phone stopped successfully")
        logger.info("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        exit(1)
