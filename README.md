# Sofia Phone

**Generic FreeSWITCH telephony integration for voice AI systems**

A reusable phone system infrastructure that works with **any** voice AI backend (hotel receptionist, restaurant booking, clinic scheduler, customer service, etc.)

---

## 🎯 Purpose

Separate the **telephony layer** (FreeSWITCH, SIP, RTP, call routing) from the **AI layer** (STT, LLM, TTS).

**This repo handles:**
- ✅ FreeSWITCH integration (SIP calls, RTP audio)
- ✅ Multi-caller session management
- ✅ Audio format conversion (8kHz telephony ↔ 16kHz/24kHz AI)
- ✅ Call features (transfer, recording, IVR, DTMF)
- ✅ Error recovery and graceful degradation

**This repo does NOT handle:**
- ❌ Speech-to-Text (your AI provides this)
- ❌ LLM/conversation logic (your AI provides this)
- ❌ Text-to-Speech (your AI provides this)

---

## 📐 Architecture

```
Incoming Call → FreeSWITCH (SIP/RTP) → sofia-phone
                                            ↓
                                    VoiceBackend Interface
                                            ↓
                                    Your AI (STT + LLM + TTS)
                                            ↓
                                    Audio Response
                                            ↓
FreeSWITCH → Caller hears AI response
```

---

## 🔌 VoiceBackend Interface

Any AI system must implement this interface:

```python
from sofia_phone.interfaces.voice_backend import VoiceBackend

class MyVoiceBackend(VoiceBackend):
    async def transcribe(self, audio: bytes, sample_rate: int) -> str:
        """Audio → Text (STT)"""
        pass

    async def generate(self, text: str, session_id: str, history: List) -> str:
        """Text → AI Response (LLM)"""
        pass

    async def speak(self, text: str) -> tuple[bytes, int]:
        """Text → Audio (TTS)"""
        pass
```

See [`src/sofia_phone/interfaces/voice_backend.py`](src/sofia_phone/interfaces/voice_backend.py) for full interface documentation.

---

## 🚀 Quick Start

### 1. Install FreeSWITCH

**macOS:**
```bash
brew install freeswitch
```

**Ubuntu/Debian:**
```bash
apt-get install freeswitch
```

### 2. Install sofia-phone

```bash
pip install -r requirements.txt
```

### 3. Implement VoiceBackend

See [`examples/`](examples/) for reference implementations.

### 4. Run

```python
from sofia_phone import PhoneHandler
from my_ai import MyVoiceBackend

backend = MyVoiceBackend()
phone = PhoneHandler(voice_backend=backend)
await phone.start()
```

---

## 📦 Project Structure

```
sofia-phone/
├── src/sofia_phone/
│   ├── interfaces/
│   │   └── voice_backend.py        # ABC interface (implement this!)
│   ├── core/
│   │   ├── phone_handler.py        # Main orchestrator
│   │   ├── session_manager.py      # Per-caller isolation
│   │   └── audio_processor.py      # Audio resampling
│   ├── freeswitch/
│   │   ├── esl_connection.py       # Event Socket Layer
│   │   ├── rtp_handler.py          # Audio streaming
│   │   └── call_control.py         # Transfer, hangup, etc.
│   └── features/
│       ├── call_recording.py       # Record all calls
│       ├── ivr.py                  # IVR menu support
│       └── error_recovery.py       # Graceful error handling
├── config/freeswitch/
│   ├── sip_profiles/               # SIP configuration
│   └── dialplan/                   # Call routing
├── tests/
│   └── mocks/
│       └── mock_voice_backend.py   # Test without real AI
├── examples/
│   └── echo_bot.py                 # Minimal example
└── scripts/
    └── install_freeswitch.sh       # Automated installation
```

---

## 🎯 Use Cases

**Hotel Reception** (sofia-ultimate):
```python
from sofia_phone import PhoneHandler
from sofia_ultimate.voice import SofiaVoiceBackend

sofia = SofiaVoiceBackend()  # Hotel AI
phone = PhoneHandler(voice_backend=sofia)
```

**Restaurant Booking**:
```python
from sofia_phone import PhoneHandler
from restaurant_ai import RestaurantVoiceBackend

restaurant = RestaurantVoiceBackend()  # Restaurant AI
phone = PhoneHandler(voice_backend=restaurant)
```

**Any Voice AI**:
```python
from sofia_phone import PhoneHandler
from your_ai import YourVoiceBackend

your_ai = YourVoiceBackend()  # Your custom AI
phone = PhoneHandler(voice_backend=your_ai)
```

---

## 🔧 Development Status

**Week 1:** Core infrastructure ⏳ (IN PROGRESS)
- [x] VoiceBackend interface defined
- [x] Repository structure created
- [x] Mock backend for testing
- [ ] FreeSWITCH installation script
- [ ] ESL connection handler
- [ ] Audio resampling layer
- [ ] RTP audio handler

**Week 2:** Production features
- [ ] Session manager (multi-caller)
- [ ] Call recording
- [ ] Call transfer
- [ ] IVR support
- [ ] Error recovery

**Week 3:** Deployment
- [ ] VPS deployment guide
- [ ] SIP trunk integration
- [ ] Monitoring/logging
- [ ] Production hardening

---

## 📝 License

MIT

---

## 🤝 Contributing

This is a reusable component. PRs welcome for:
- New VoiceBackend implementations
- Additional call features
- Bug fixes
- Documentation improvements

---

## 📚 Related Projects

- [sofia-ultimate](https://github.com/ezekaj/elo-sofia-larchemont-hotel) - Hotel receptionist AI (uses sofia-phone)

---

Built with ❤️ for production voice AI telephony
