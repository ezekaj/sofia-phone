# Twilio Integration Guide for Sofia-Phone

## Quick Setup (15 minutes)

### 1. Get Twilio Account
1. Sign up at https://www.twilio.com/try-twilio
2. Get a free trial account ($15 credit)
3. Note your **Account SID** and **Auth Token**

### 2. Buy a Phone Number
```bash
# Or use Twilio Console: https://console.twilio.com/us1/develop/phone-numbers/manage/incoming
# Cost: $1/month
```

### 3. Install Twilio SDK
```bash
cd /Users/tolga/Desktop/sofia-phone
source .venv/bin/activate
pip install twilio
```

### 4. Create Twilio Handler

Create `src/sofia_phone/integrations/twilio_handler.py`:

```python
"""Twilio integration for Sofia-Phone"""
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect
import os

class TwilioHandler:
    def __init__(self, account_sid: str, auth_token: str):
        self.client = Client(account_sid, auth_token)

    def configure_phone_number(self, phone_number: str, voice_url: str):
        """
        Configure Twilio phone number to connect to sofia-phone

        phone_number: Your Twilio number (e.g., "+15551234567")
        voice_url: Public URL where sofia-phone is running
        """
        incoming_phone_numbers = self.client.incoming_phone_numbers.list(
            phone_number=phone_number
        )

        if incoming_phone_numbers:
            number = incoming_phone_numbers[0]
            number.update(
                voice_url=voice_url,
                voice_method='POST'
            )
            return f"Configured {phone_number} -> {voice_url}"
        else:
            raise ValueError(f"Phone number {phone_number} not found in account")

    @staticmethod
    def generate_twiml_for_sofia_phone(websocket_url: str) -> str:
        """
        Generate TwiML to connect incoming call to sofia-phone via WebSocket

        websocket_url: ws://your-server.com:8084
        """
        response = VoiceResponse()
        connect = Connect()
        connect.stream(url=websocket_url)
        response.append(connect)
        return str(response)
```

### 5. Expose Sofia-Phone to Internet

**Option A: ngrok (Fastest - for testing)**
```bash
# Install ngrok
brew install ngrok

# Start sofia-phone
cd /Users/tolga/Desktop/sofia-phone
source .venv/bin/activate
python -m sofia_phone

# In another terminal, expose port 8084
ngrok tcp 8084

# Note the ngrok URL (e.g., tcp://0.tcp.ngrok.io:12345)
```

**Option B: Deploy to Cloud (Production)**
```bash
# Use the Dockerfile provided
docker build -t sofia-phone .
docker run -p 8084:8084 sofia-phone

# Deploy to:
# - AWS ECS/EC2
# - Google Cloud Run
# - DigitalOcean
# - Fly.io
```

### 6. Create Twilio Webhook Endpoint

Add to `src/sofia_phone/__main__.py`:

```python
from aiohttp import web

async def twilio_webhook(request):
    """Handle incoming Twilio calls"""
    # Get call SID from Twilio
    call_sid = request.rel_url.query.get('CallSid')

    # Generate TwiML to connect to sofia-phone WebSocket
    twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://your-ngrok-url.ngrok.io/stream"/>
    </Connect>
</Response>'''

    return web.Response(text=twiml, content_type='text/xml')

# Add route
app.router.add_post('/twilio/voice', twilio_webhook)
```

### 7. Configure Twilio Number

```python
from sofia_phone.integrations.twilio_handler import TwilioHandler

handler = TwilioHandler(
    account_sid='YOUR_ACCOUNT_SID',
    auth_token='YOUR_AUTH_TOKEN'
)

handler.configure_phone_number(
    phone_number='+15551234567',  # Your Twilio number
    voice_url='https://your-ngrok-url.ngrok.io/twilio/voice'
)
```

## Architecture

```
Phone Call → Twilio Number → Webhook → Sofia-Phone
                                       ↓
                           ESL Handler (port 8084)
                                       ↓
                           Memory Backend
                                       ↓
                           STT → LLM → TTS
                                       ↓
                           Response to Caller
```

## Testing

1. **Start sofia-phone:**
```bash
cd /Users/tolga/Desktop/sofia-phone
source .venv/bin/activate
python -m sofia_phone
```

2. **Expose with ngrok:**
```bash
ngrok http 8084
```

3. **Configure Twilio** (use ngrok URL)

4. **Call your Twilio number** from any phone

5. **Check logs:**
```bash
tail -f logs/sofia-phone.log
```

## Production Checklist

- [ ] Deploy sofia-phone to cloud (not localhost)
- [ ] Use HTTPS for webhook endpoint
- [ ] Set up proper auth for Twilio webhooks
- [ ] Configure error handling
- [ ] Set up monitoring (health checks working on port 8080)
- [ ] Test with real phone calls (not just Zoiper)
- [ ] Verify memory system works across calls
- [ ] Set up call recording (optional)

## Costs

- **Twilio Phone Number:** $1/month
- **Incoming Calls:** $0.0085/minute
- **Outgoing Calls:** $0.013/minute
- **Cloud Hosting:** ~$5-20/month (DigitalOcean Droplet)

**Total for testing:** ~$1-2 for first month

## Next Steps

1. Get Twilio account
2. Buy phone number
3. Deploy sofia-phone (ngrok for testing)
4. Configure webhook
5. Make test call
6. Verify memory works across multiple calls

## Support

- Twilio Console: https://console.twilio.com
- Twilio Docs: https://www.twilio.com/docs/voice
- Sofia-Phone Issues: Check logs/sofia-phone.log
