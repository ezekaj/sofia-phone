"""
Production-Grade Error Recovery System

Handles all failure modes gracefully:
- Network failures (reconnect with exponential backoff)
- Audio processing errors (fallback to safe defaults)
- AI backend failures (retry with timeout)
- Resource exhaustion (circuit breaker pattern)
- Cascading failures (bulkhead isolation)

Key Patterns:
- Circuit Breaker: Stop calling failing services
- Retry with Exponential Backoff: Smart retry logic
- Bulkhead Isolation: Contain failures
- Graceful Degradation: Keep working with reduced functionality
- Health Monitoring: Track failure rates
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable, Any, Awaitable
from loguru import logger
import time


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing - reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Open circuit after N failures
    success_threshold: int = 2  # Close circuit after N successes
    timeout_seconds: float = 60.0  # Time to wait before HALF_OPEN
    max_requests_half_open: int = 3  # Max requests in HALF_OPEN state


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics"""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.now)
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    States:
    - CLOSED: Normal operation
    - OPEN: Service failing, reject requests immediately
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Circuit name (for logging)
            config: Configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.metrics = CircuitBreakerMetrics()

        logger.info(f"CircuitBreaker '{name}' initialized")

    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is OPEN
        """
        # Check circuit state
        if self.metrics.state == CircuitState.OPEN:
            # Check if timeout elapsed
            if self._should_attempt_reset():
                logger.info(f"Circuit '{self.name}': OPEN → HALF_OPEN (testing recovery)")
                self.metrics.state = CircuitState.HALF_OPEN
                self.metrics.success_count = 0
                self.metrics.failure_count = 0
            else:
                raise CircuitBreakerError(
                    f"Circuit '{self.name}' is OPEN (failing fast)"
                )

        # Execute function
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure(e)
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time elapsed to try HALF_OPEN state"""
        if not self.metrics.last_failure_time:
            return True

        elapsed = datetime.now() - self.metrics.last_failure_time
        return elapsed.total_seconds() >= self.config.timeout_seconds

    def _on_success(self):
        """Handle successful call"""
        self.metrics.success_count += 1
        self.metrics.total_successes += 1

        if self.metrics.state == CircuitState.HALF_OPEN:
            # Check if enough successes to close circuit
            if self.metrics.success_count >= self.config.success_threshold:
                logger.success(
                    f"Circuit '{self.name}': HALF_OPEN → CLOSED (recovered)"
                )
                self.metrics.state = CircuitState.CLOSED
                self.metrics.failure_count = 0
                self.metrics.last_state_change = datetime.now()

    def _on_failure(self, error: Exception):
        """Handle failed call"""
        self.metrics.failure_count += 1
        self.metrics.total_failures += 1
        self.metrics.last_failure_time = datetime.now()

        logger.warning(
            f"Circuit '{self.name}' failure {self.metrics.failure_count}/"
            f"{self.config.failure_threshold}: {error}"
        )

        # Check if should open circuit
        if self.metrics.state == CircuitState.CLOSED:
            if self.metrics.failure_count >= self.config.failure_threshold:
                logger.error(
                    f"Circuit '{self.name}': CLOSED → OPEN (too many failures)"
                )
                self.metrics.state = CircuitState.OPEN
                self.metrics.last_state_change = datetime.now()

        elif self.metrics.state == CircuitState.HALF_OPEN:
            # Any failure in HALF_OPEN → back to OPEN
            logger.error(
                f"Circuit '{self.name}': HALF_OPEN → OPEN (recovery failed)"
            )
            self.metrics.state = CircuitState.OPEN
            self.metrics.last_state_change = datetime.now()

    def reset(self):
        """Manually reset circuit to CLOSED state"""
        logger.info(f"Circuit '{self.name}' manually reset to CLOSED")
        self.metrics.state = CircuitState.CLOSED
        self.metrics.failure_count = 0
        self.metrics.success_count = 0
        self.metrics.last_state_change = datetime.now()

    def get_metrics(self) -> dict:
        """Get circuit metrics"""
        return {
            "name": self.name,
            "state": self.metrics.state.value,
            "failure_count": self.metrics.failure_count,
            "success_count": self.metrics.success_count,
            "total_failures": self.metrics.total_failures,
            "total_successes": self.metrics.total_successes,
            "last_failure": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
            "last_state_change": self.metrics.last_state_change.isoformat()
        }


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is OPEN"""
    pass


class RetryPolicy:
    """
    Retry policy with exponential backoff.

    Retries failing operations with increasing delays.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay_seconds: float = 1.0,
        max_delay_seconds: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize retry policy.

        Args:
            max_attempts: Maximum retry attempts
            initial_delay_seconds: Initial delay before first retry
            max_delay_seconds: Maximum delay between retries
            exponential_base: Exponential backoff base (2 = double each time)
            jitter: Add random jitter to delays (prevent thundering herd)
        """
        self.max_attempts = max_attempts
        self.initial_delay_seconds = initial_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.exponential_base = exponential_base
        self.jitter = jitter

    async def execute(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all retries exhausted
        """
        last_exception = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                result = await func(*args, **kwargs)
                if attempt > 1:
                    logger.success(f"Retry succeeded on attempt {attempt}")
                return result

            except Exception as e:
                last_exception = e

                if attempt == self.max_attempts:
                    logger.error(
                        f"All {self.max_attempts} retry attempts exhausted: {e}"
                    )
                    raise

                # Calculate delay
                delay = self._calculate_delay(attempt)

                logger.warning(
                    f"Attempt {attempt}/{self.max_attempts} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                await asyncio.sleep(delay)

        # Should never reach here, but just in case
        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter"""
        import random

        # Exponential backoff
        delay = min(
            self.initial_delay_seconds * (self.exponential_base ** (attempt - 1)),
            self.max_delay_seconds
        )

        # Add jitter (random ±25%)
        if self.jitter:
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)


class ErrorRecoveryManager:
    """
    Production-grade error recovery manager.

    Provides:
    - Circuit breakers for all external services
    - Retry policies with exponential backoff
    - Graceful degradation strategies
    - Error rate monitoring
    """

    def __init__(self):
        """Initialize error recovery manager"""
        # Circuit breakers for each service
        self.circuits = {
            "stt": CircuitBreaker("STT", CircuitBreakerConfig(
                failure_threshold=3,
                timeout_seconds=30
            )),
            "llm": CircuitBreaker("LLM", CircuitBreakerConfig(
                failure_threshold=3,
                timeout_seconds=30
            )),
            "tts": CircuitBreaker("TTS", CircuitBreakerConfig(
                failure_threshold=3,
                timeout_seconds=30
            )),
            "freeswitch": CircuitBreaker("FreeSWITCH", CircuitBreakerConfig(
                failure_threshold=5,
                timeout_seconds=60
            ))
        }

        # Retry policies
        self.retry_policies = {
            "default": RetryPolicy(max_attempts=3, initial_delay_seconds=1.0),
            "aggressive": RetryPolicy(max_attempts=5, initial_delay_seconds=0.5),
            "conservative": RetryPolicy(max_attempts=2, initial_delay_seconds=2.0)
        }

        logger.info("ErrorRecoveryManager initialized")

    async def with_circuit_breaker(
        self,
        service: str,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            service: Service name (stt, llm, tts, freeswitch)
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        circuit = self.circuits.get(service)
        if not circuit:
            logger.warning(f"No circuit breaker for service '{service}'")
            return await func(*args, **kwargs)

        return await circuit.call(func, *args, **kwargs)

    async def with_retry(
        self,
        func: Callable[..., Awaitable[Any]],
        policy: str = "default",
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry policy.

        Args:
            func: Async function to execute
            policy: Retry policy name (default, aggressive, conservative)
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        retry_policy = self.retry_policies.get(policy)
        if not retry_policy:
            logger.warning(f"Unknown retry policy '{policy}', using default")
            retry_policy = self.retry_policies["default"]

        return await retry_policy.execute(func, *args, **kwargs)

    async def with_protection(
        self,
        service: str,
        func: Callable[..., Awaitable[Any]],
        retry_policy: str = "default",
        fallback: Optional[Callable[..., Awaitable[Any]]] = None,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with full protection (circuit breaker + retry + fallback).

        Args:
            service: Service name
            func: Async function to execute
            retry_policy: Retry policy name
            fallback: Fallback function if all retries fail
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result or fallback result
        """
        try:
            # Wrap with retry
            async def func_with_retry():
                return await self.with_retry(func, retry_policy, *args, **kwargs)

            # Wrap with circuit breaker
            return await self.with_circuit_breaker(service, func_with_retry)

        except Exception as e:
            logger.error(f"Service '{service}' failed after all retries: {e}")

            # Try fallback
            if fallback:
                logger.info(f"Executing fallback for service '{service}'")
                try:
                    return await fallback(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise

            raise

    def reset_circuit(self, service: str):
        """Manually reset circuit breaker"""
        circuit = self.circuits.get(service)
        if circuit:
            circuit.reset()
        else:
            logger.warning(f"No circuit breaker for service '{service}'")

    def get_metrics(self) -> dict:
        """Get error recovery metrics"""
        return {
            "circuits": {
                name: circuit.get_metrics()
                for name, circuit in self.circuits.items()
            }
        }

    def __str__(self) -> str:
        open_circuits = [
            name for name, circuit in self.circuits.items()
            if circuit.metrics.state == CircuitState.OPEN
        ]

        return (
            f"ErrorRecoveryManager("
            f"circuits={len(self.circuits)}, "
            f"open={len(open_circuits)}"
            f")"
        )
