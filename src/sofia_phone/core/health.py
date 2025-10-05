"""
Production-Grade Health Check System

Provides:
- HTTP health endpoints (/health, /ready, /metrics)
- Component health checks
- System metrics
- Kubernetes-compatible liveness/readiness probes

Endpoints:
- GET /health - Overall system health
- GET /ready - Readiness for traffic
- GET /metrics - Prometheus-compatible metrics
- GET /status - Detailed status
"""
import asyncio
from aiohttp import web
from typing import Dict, Any, Optional, Callable, Awaitable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from loguru import logger
import psutil
import time


class HealthStatus(Enum):
    """Health status values"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status for a component"""
    name: str
    status: HealthStatus
    message: str = ""
    last_check: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """
    Health check system.

    Performs periodic health checks on all components
    and exposes HTTP endpoints for monitoring.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        check_interval_seconds: int = 30
    ):
        """
        Initialize health checker.

        Args:
            host: HTTP server host
            port: HTTP server port
            check_interval_seconds: Health check frequency
        """
        self.host = host
        self.port = port
        self.check_interval_seconds = check_interval_seconds

        # Component health checks
        self._health_checks: Dict[str, Callable[[], Awaitable[ComponentHealth]]] = {}
        self._component_health: Dict[str, ComponentHealth] = {}

        # System metrics
        self._start_time = datetime.now()
        self._request_count = 0
        self._error_count = 0

        # Background tasks
        self._check_task: Optional[asyncio.Task] = None
        self._running = False

        # HTTP server
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

        logger.info(f"HealthChecker initialized on {host}:{port}")

    def register_check(
        self,
        name: str,
        check_func: Callable[[], Awaitable[ComponentHealth]]
    ):
        """
        Register component health check.

        Args:
            name: Component name
            check_func: Async function that returns ComponentHealth
        """
        self._health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")

    async def start(self):
        """Start health check system"""
        if self._running:
            return

        self._running = True

        # Setup HTTP server
        self._app = web.Application()
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/ready", self._handle_ready)
        self._app.router.add_get("/metrics", self._handle_metrics)
        self._app.router.add_get("/status", self._handle_status)

        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        # Start background health checks
        self._check_task = asyncio.create_task(self._check_loop())

        logger.success(f"Health check server started on http://{self.host}:{self.port}")
        logger.info(
            f"Endpoints: /health (liveness), /ready (readiness), "
            f"/metrics (prometheus), /status (detailed)"
        )

    async def stop(self):
        """Stop health check system"""
        self._running = False

        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

        if self._site:
            await self._site.stop()

        if self._runner:
            await self._runner.cleanup()

        logger.info("Health check server stopped")

    async def _check_loop(self):
        """Background task to run health checks"""
        while self._running:
            try:
                await self._run_health_checks()
                await asyncio.sleep(self.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.check_interval_seconds)

    async def _run_health_checks(self):
        """Run all registered health checks"""
        for name, check_func in self._health_checks.items():
            try:
                health = await check_func()
                self._component_health[name] = health

                if health.status == HealthStatus.UNHEALTHY:
                    logger.warning(f"Component '{name}' is unhealthy: {health.message}")
                elif health.status == HealthStatus.DEGRADED:
                    logger.info(f"Component '{name}' is degraded: {health.message}")

            except Exception as e:
                logger.error(f"Health check failed for '{name}': {e}")
                self._component_health[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check error: {e}"
                )

    async def _handle_health(self, request: web.Request) -> web.Response:
        """
        Handle /health endpoint (liveness probe).

        Returns 200 if system is alive, 503 if unhealthy.
        """
        self._request_count += 1

        # Check overall health
        overall_status = self._get_overall_status()

        if overall_status == HealthStatus.HEALTHY:
            return web.json_response({
                "status": "healthy",
                "timestamp": datetime.now().isoformat()
            }, status=200)
        else:
            return web.json_response({
                "status": overall_status.value,
                "timestamp": datetime.now().isoformat()
            }, status=503)

    async def _handle_ready(self, request: web.Request) -> web.Response:
        """
        Handle /ready endpoint (readiness probe).

        Returns 200 if system is ready for traffic, 503 if not.
        """
        self._request_count += 1

        # Check if all components are healthy or degraded (not unhealthy)
        ready = all(
            health.status != HealthStatus.UNHEALTHY
            for health in self._component_health.values()
        )

        if ready:
            return web.json_response({
                "status": "ready",
                "timestamp": datetime.now().isoformat()
            }, status=200)
        else:
            unhealthy = [
                name for name, health in self._component_health.items()
                if health.status == HealthStatus.UNHEALTHY
            ]

            return web.json_response({
                "status": "not_ready",
                "unhealthy_components": unhealthy,
                "timestamp": datetime.now().isoformat()
            }, status=503)

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        """
        Handle /metrics endpoint (Prometheus-compatible).

        Returns metrics in Prometheus text format.
        """
        self._request_count += 1

        # System metrics
        uptime = (datetime.now() - self._start_time).total_seconds()
        memory = psutil.Process().memory_info()
        cpu_percent = psutil.Process().cpu_percent()

        # Build Prometheus metrics
        metrics = []

        # Uptime
        metrics.append(f"sofia_phone_uptime_seconds {uptime}")

        # Requests
        metrics.append(f"sofia_phone_http_requests_total {self._request_count}")
        metrics.append(f"sofia_phone_http_errors_total {self._error_count}")

        # Memory
        metrics.append(f"sofia_phone_memory_rss_bytes {memory.rss}")
        metrics.append(f"sofia_phone_memory_vms_bytes {memory.vms}")

        # CPU
        metrics.append(f"sofia_phone_cpu_percent {cpu_percent}")

        # Component health (1=healthy, 0.5=degraded, 0=unhealthy)
        for name, health in self._component_health.items():
            value = {
                HealthStatus.HEALTHY: 1.0,
                HealthStatus.DEGRADED: 0.5,
                HealthStatus.UNHEALTHY: 0.0
            }[health.status]

            safe_name = name.replace("-", "_").replace(" ", "_")
            metrics.append(f"sofia_phone_component_health{{component=\"{safe_name}\"}} {value}")

        return web.Response(
            text="\n".join(metrics) + "\n",
            content_type="text/plain"
        )

    async def _handle_status(self, request: web.Request) -> web.Response:
        """
        Handle /status endpoint (detailed status).

        Returns detailed status of all components.
        """
        self._request_count += 1

        uptime = (datetime.now() - self._start_time).total_seconds()
        memory = psutil.Process().memory_info()

        return web.json_response({
            "status": self._get_overall_status().value,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "system": {
                "memory_rss_mb": memory.rss / (1024 * 1024),
                "memory_vms_mb": memory.vms / (1024 * 1024),
                "cpu_percent": psutil.Process().cpu_percent()
            },
            "components": {
                name: {
                    "status": health.status.value,
                    "message": health.message,
                    "last_check": health.last_check.isoformat(),
                    "metadata": health.metadata
                }
                for name, health in self._component_health.items()
            },
            "metrics": {
                "http_requests": self._request_count,
                "http_errors": self._error_count
            }
        })

    def _get_overall_status(self) -> HealthStatus:
        """Get overall system health status"""
        if not self._component_health:
            return HealthStatus.HEALTHY

        # If any component is unhealthy → system is unhealthy
        if any(h.status == HealthStatus.UNHEALTHY for h in self._component_health.values()):
            return HealthStatus.UNHEALTHY

        # If any component is degraded → system is degraded
        if any(h.status == HealthStatus.DEGRADED for h in self._component_health.values()):
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY


# Built-in health checks

async def check_freeswich_health(phone_handler) -> ComponentHealth:
    """
    Check FreeSWITCH connectivity.

    Args:
        phone_handler: PhoneHandler instance

    Returns:
        ComponentHealth
    """
    try:
        # Check if ESL server is running
        if phone_handler._esl_server and phone_handler._running:
            return ComponentHealth(
                name="freeswitch",
                status=HealthStatus.HEALTHY,
                message="ESL server running",
                metadata={
                    "esl_port": phone_handler.esl_port,
                    "active_calls": len(phone_handler._active_calls)
                }
            )
        else:
            return ComponentHealth(
                name="freeswitch",
                status=HealthStatus.UNHEALTHY,
                message="ESL server not running"
            )

    except Exception as e:
        return ComponentHealth(
            name="freeswitch",
            status=HealthStatus.UNHEALTHY,
            message=f"Health check failed: {e}"
        )


async def check_session_manager_health(session_manager) -> ComponentHealth:
    """
    Check session manager health.

    Args:
        session_manager: SessionManager instance

    Returns:
        ComponentHealth
    """
    try:
        metrics = session_manager.get_metrics()
        active = metrics["active_sessions"]
        max_sessions = metrics["max_sessions"]

        # Degraded if >80% capacity
        if active > max_sessions * 0.8:
            return ComponentHealth(
                name="session_manager",
                status=HealthStatus.DEGRADED,
                message=f"High capacity: {active}/{max_sessions} sessions",
                metadata=metrics
            )

        return ComponentHealth(
            name="session_manager",
            status=HealthStatus.HEALTHY,
            message=f"{active}/{max_sessions} sessions active",
            metadata=metrics
        )

    except Exception as e:
        return ComponentHealth(
            name="session_manager",
            status=HealthStatus.UNHEALTHY,
            message=f"Health check failed: {e}"
        )


async def check_voice_backend_health(voice_backend) -> ComponentHealth:
    """
    Check voice backend health.

    Args:
        voice_backend: VoiceBackend instance

    Returns:
        ComponentHealth
    """
    try:
        # Quick health check (generate empty response)
        # This tests LLM connectivity
        await asyncio.wait_for(
            voice_backend.generate("", session_id="health-check", history=[]),
            timeout=5.0
        )

        return ComponentHealth(
            name="voice_backend",
            status=HealthStatus.HEALTHY,
            message="Voice backend responsive"
        )

    except asyncio.TimeoutError:
        return ComponentHealth(
            name="voice_backend",
            status=HealthStatus.DEGRADED,
            message="Voice backend slow (>5s)"
        )
    except Exception as e:
        return ComponentHealth(
            name="voice_backend",
            status=HealthStatus.UNHEALTHY,
            message=f"Voice backend error: {e}"
        )
