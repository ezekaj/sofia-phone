# Sofia-Phone

**Production-Ready Telephony Layer for Voice AI Applications**

Sofia-phone is a robust, scalable telephony infrastructure built on FreeSWITCH. It provides a clean interface for integrating any voice AI backend with real phone systems.

## 🎯 Key Features

- **Production-Grade Architecture**: Built for reliability, scalability, and monitoring
- **FreeSWITCH Integration**: Full SIP/RTP support via Event Socket Layer (ESL)
- **Multi-Session Management**: Handle 100+ concurrent calls with session isolation
- **Professional Audio Processing**: High-quality resampling (8kHz ↔ 16kHz ↔ 24kHz)
- **Error Recovery**: Circuit breakers, retry policies, graceful degradation
- **Health Monitoring**: Kubernetes-ready liveness/readiness probes
- **Docker Deployment**: Complete containerized setup
- **Pluggable AI Backend**: Clean interface for any STT/LLM/TTS stack

## 🏗️ Architecture

```
┌─────────────┐     SIP      ┌──────────────┐     ESL/RTP    ┌──────────────┐
│ Phone/SIP   │◄────────────►│ FreeSWITCH   │◄──────────────►│ sofia-phone  │
│ Client      │              │              │                 │              │
└─────────────┘              └──────────────┘                 └──────┬───────┘
                                                                     │
                                                       ┌─────────────┼─────────────┐
                                                       │             │             │
                                                 ┌─────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
                                                 │  Session  │ │  Audio  │ │   Voice   │
                                                 │  Manager  │ │Processor│ │  Backend  │
                                                 └───────────┘ └─────────┘ └───────────┘
                                                                                   │
                                                                            ┌──────┼──────┐
                                                                       ┌────▼────┐ │ ┌────▼────┐
                                                                       │ Whisper │ │ │  Edge   │
                                                                       │  (STT)  │ │ │  (TTS)  │
                                                                       └─────────┘ │ └─────────┘
                                                                             ┌─────▼─────┐
                                                                             │  Ollama   │
                                                                             │  (LLM)    │
                                                                             └───────────┘
```

## 📦 Components

### Core Components

- **PhoneHandler**: Main orchestrator - manages call flow
- **SessionManager**: Thread-safe multi-session management
- **AudioProcessor**: Professional-grade audio resampling
- **RTPHandler**: Real-time audio streaming
- **ESLConnection**: FreeSWITCH Event Socket Layer integration

### Production Infrastructure

- **ErrorRecoveryManager**: Circuit breakers, retries, fallbacks
- **HealthChecker**: HTTP health endpoints + Prometheus metrics
- **Config**: Environment-based configuration
- **Logging**: Structured logging (console + file + JSON)

### Interface

- **VoiceBackend**: Abstract interface for AI integration
  - `transcribe()`: Speech-to-text
  - `generate()`: LLM response generation
  - `speak()`: Text-to-speech

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- FreeSWITCH 1.10+ (or use Docker)
- SIP softphone (Zoiper, Linphone, etc.)

### Local Development (Mac)

1. **Clone the repository**
```bash
cd ~/Desktop
git clone git@github.com:ezekaj/sofia-phone.git
cd sofia-phone
```

2. **Create virtual environment**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. **Install FreeSWITCH** (Mac)
```bash
./scripts/install_freeswitch_mac.sh
```

4. **Configure FreeSWITCH**
```bash
./scripts/setup_freeswitch.sh
```

5. **Start FreeSWITCH**
```bash
brew services start freeswitch
```

6. **Run sofia-phone** (with mock backend for testing)
```bash
python -m sofia_phone
```

7. **Make a test call**
- Configure your SIP client (Zoiper):
  - Server: `127.0.0.1:5060`
  - No authentication
- Call any number (e.g., `1000`)
- sofia-phone will answer with mock AI

### Docker Deployment (Production)

1. **Build and start** (includes FreeSWITCH)
```bash
docker-compose up -d
```

2. **View logs**
```bash
docker-compose logs -f sofia-phone
```

3. **Health check**
```bash
curl http://localhost:8080/health
curl http://localhost:8080/ready
curl http://localhost:8080/metrics
curl http://localhost:8080/status
```

4. **Stop**
```bash
docker-compose down
```

### With Monitoring (Prometheus + Grafana)

```bash
docker-compose --profile monitoring up -d
```

- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

## 🔌 Integrating Your AI Backend

Implement the `VoiceBackend` interface:

```python
from sofia_phone.interfaces.voice_backend import VoiceBackend
from typing import List, Dict, Optional, Tuple

class MyVoiceBackend(VoiceBackend):
    async def transcribe(self, audio: bytes, sample_rate: int = 8000) -> str:
        """
        Speech-to-Text

        Args:
            audio: Raw PCM audio (mono, 16-bit)
            sample_rate: Sample rate in Hz

        Returns:
            Transcribed text
        """
        # Your STT implementation (Whisper, etc.)
        pass

    async def generate(
        self,
        text: str,
        session_id: str,
        history: Optional[List[Dict]] = None
    ) -> str:
        """
        LLM Response Generation

        Args:
            text: User input text
            session_id: Session identifier
            history: Conversation history

        Returns:
            AI response text
        """
        # Your LLM implementation (Ollama, OpenAI, etc.)
        pass

    async def speak(self, text: str) -> Tuple[bytes, int]:
        """
        Text-to-Speech

        Args:
            text: Text to speak

        Returns:
            (audio_bytes, sample_rate)
        """
        # Your TTS implementation (Edge TTS, etc.)
        pass
```

Then inject it:

```python
from sofia_phone.core.phone_handler import PhoneHandler

backend = MyVoiceBackend()
handler = PhoneHandler(voice_backend=backend)
await handler.start()
```

## 📊 Monitoring

### Health Endpoints

- **GET /health** - Liveness probe (200 = alive)
- **GET /ready** - Readiness probe (200 = ready for traffic)
- **GET /metrics** - Prometheus metrics
- **GET /status** - Detailed status JSON

### Key Metrics

```
sofia_phone_uptime_seconds
sofia_phone_http_requests_total
sofia_phone_memory_rss_bytes
sofia_phone_cpu_percent
sofia_phone_component_health{component="..."}
```

### Logs

- Console: Pretty-printed with colors (development)
- File: `/app/logs/sofia-phone.log` (production)
- Format: JSON in production, human-readable in dev
- Rotation: 100 MB
- Retention: 1 week

## ⚙️ Configuration

All settings via environment variables:

### General
```bash
SOFIA_PHONE_ENV=production  # development/production
SOFIA_PHONE_DEBUG=false
```

### FreeSWITCH
```bash
SOFIA_PHONE_ESL_HOST=0.0.0.0
SOFIA_PHONE_ESL_PORT=8084
SOFIA_PHONE_RTP_START_PORT=16384
```

### Sessions
```bash
SOFIA_PHONE_MAX_SESSIONS=100
SOFIA_PHONE_SESSION_TIMEOUT=30  # minutes
```

### Logging
```bash
SOFIA_PHONE_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
SOFIA_PHONE_LOG_FILE=/app/logs/sofia-phone.log
SOFIA_PHONE_JSON_LOGS=true  # JSON format (production)
```

### Health Checks
```bash
SOFIA_PHONE_HEALTH_CHECK=true
SOFIA_PHONE_HEALTH_PORT=8080
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_audio_processor.py

# Run with coverage
pytest --cov=sofia_phone --cov-report=html

# View coverage
open htmlcov/index.html
```

## 📁 Project Structure

```
sofia-phone/
├── src/
│   └── sofia_phone/
│       ├── __main__.py          # Main entry point
│       ├── interfaces/
│       │   └── voice_backend.py # AI backend interface
│       ├── core/
│       │   ├── phone_handler.py # Main orchestrator
│       │   ├── session_manager.py
│       │   ├── audio_processor.py
│       │   ├── error_recovery.py
│       │   ├── config.py
│       │   ├── logging_setup.py
│       │   └── health.py
│       └── freeswitch/
│           ├── esl_connection.py
│           └── rtp_handler.py
├── tests/
│   ├── test_audio_processor.py
│   ├── test_session_manager.py
│   ├── test_integration.py
│   └── mocks/
│       └── mock_voice_backend.py
├── config/
│   └── freeswitch/
│       ├── sip_profiles/internal.xml
│       └── dialplan/sofia_phone.xml
├── scripts/
│   ├── install_freeswitch_mac.sh
│   └── setup_freeswitch.sh
├── Dockerfile
├── docker-compose.yml
├── docker-entrypoint.sh
├── requirements.txt
└── README.md
```

## 🔒 Production Checklist

- [x] Error recovery (circuit breakers, retries)
- [x] Health monitoring (liveness, readiness)
- [x] Structured logging (JSON format)
- [x] Graceful shutdown (SIGTERM handling)
- [x] Configuration via environment
- [x] Docker containerization
- [x] Multi-session isolation
- [x] Audio quality validation
- [x] Comprehensive tests
- [x] Prometheus metrics
- [ ] TLS/SRTP encryption
- [ ] Rate limiting
- [ ] Authentication/authorization

## 🐛 Troubleshooting

### FreeSWITCH not starting

```bash
# Check status
brew services list | grep freeswitch

# View logs
tail -f /opt/homebrew/var/log/freeswitch/freeswitch.log

# Restart
brew services restart freeswitch
```

### No audio in calls

```bash
# Check RTP ports are open
netstat -an | grep 16384

# Check audio processing logs
tail -f logs/sofia-phone.log | grep -i audio

# Test with echo extension (dial 9999)
```

### Connection refused to FreeSWITCH

```bash
# Check ESL is listening
netstat -an | grep 8084

# Check FreeSWITCH ESL config
cat /opt/homebrew/etc/freeswitch/autoload_configs/event_socket.conf.xml
```

## 📄 License

MIT License - See LICENSE file

## 🤝 Contributing

This is a reusable telephony layer. Contributions welcome:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

## 🙏 Acknowledgments

- FreeSWITCH - Open source telephony platform
- scipy - Scientific audio processing
- loguru - Beautiful logging
- aiohttp - Async HTTP server

## 📞 Support

For issues or questions:
- GitHub Issues: https://github.com/ezekaj/sofia-phone/issues

---

**Built with ❤️ for the voice AI community**
