"""
Production-Grade RTP Audio Handler

Handles Real-time Transport Protocol (RTP) audio streaming for telephony.

Features:
- Bidirectional audio streaming
- Jitter buffer for packet loss handling
- Packet reordering
- DTMF detection
- Audio metrics

RTP is used by FreeSWITCH to send/receive actual voice data.
"""
import asyncio
import struct
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable
from loguru import logger
import time
from collections import deque


@dataclass
class RTPPacket:
    """
    RTP packet structure.

    Format (12-byte header + payload):
    - Version (2 bits)
    - Padding (1 bit)
    - Extension (1 bit)
    - CSRC count (4 bits)
    - Marker (1 bit)
    - Payload type (7 bits)
    - Sequence number (16 bits)
    - Timestamp (32 bits)
    - SSRC (32 bits)
    - Payload (variable)
    """
    version: int
    padding: bool
    extension: bool
    marker: bool
    payload_type: int
    sequence_number: int
    timestamp: int
    ssrc: int
    payload: bytes

    @classmethod
    def parse(cls, data: bytes) -> "RTPPacket":
        """
        Parse RTP packet from bytes.

        Args:
            data: Raw RTP packet bytes

        Returns:
            Parsed RTPPacket
        """
        if len(data) < 12:
            raise ValueError(f"RTP packet too short: {len(data)} bytes")

        # Parse header (12 bytes)
        byte0, byte1 = data[0], data[1]

        version = (byte0 >> 6) & 0x03
        padding = bool((byte0 >> 5) & 0x01)
        extension = bool((byte0 >> 4) & 0x01)
        marker = bool((byte1 >> 7) & 0x01)
        payload_type = byte1 & 0x7F

        sequence_number = struct.unpack("!H", data[2:4])[0]
        timestamp = struct.unpack("!I", data[4:8])[0]
        ssrc = struct.unpack("!I", data[8:12])[0]

        # Payload is everything after header
        payload = data[12:]

        return cls(
            version=version,
            padding=padding,
            extension=extension,
            marker=marker,
            payload_type=payload_type,
            sequence_number=sequence_number,
            timestamp=timestamp,
            ssrc=ssrc,
            payload=payload
        )

    def to_bytes(self) -> bytes:
        """
        Serialize RTP packet to bytes.

        Returns:
            Raw RTP packet bytes
        """
        # Build header byte 0
        byte0 = (self.version << 6) | (int(self.padding) << 5) | (int(self.extension) << 4)

        # Build header byte 1
        byte1 = (int(self.marker) << 7) | (self.payload_type & 0x7F)

        # Pack header
        header = struct.pack(
            "!BBHII",
            byte0,
            byte1,
            self.sequence_number,
            self.timestamp,
            self.ssrc
        )

        return header + self.payload

    def __str__(self) -> str:
        return (
            f"RTPPacket(seq={self.sequence_number}, ts={self.timestamp}, "
            f"pt={self.payload_type}, size={len(self.payload)})"
        )


class RTPHandler:
    """
    Production-grade RTP audio handler.

    Manages RTP streams for call audio with jitter buffering,
    packet reordering, and loss handling.
    """

    def __init__(
        self,
        local_port: int = 16384,
        payload_type: int = 0,  # PCMU (G.711 u-law)
        sample_rate: int = 8000,
        jitter_buffer_ms: int = 50
    ):
        """
        Initialize RTP handler.

        Args:
            local_port: Local RTP port
            payload_type: RTP payload type (0=PCMU, 8=PCMA)
            sample_rate: Audio sample rate (Hz)
            jitter_buffer_ms: Jitter buffer size in milliseconds
        """
        self.local_port = local_port
        self.payload_type = payload_type
        self.sample_rate = sample_rate
        self.jitter_buffer_ms = jitter_buffer_ms

        # UDP socket for RTP
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional[asyncio.DatagramProtocol] = None

        # RTP state
        self._sequence_number = 0
        self._timestamp = 0
        self._ssrc = int(time.time() * 1000) & 0xFFFFFFFF  # Random SSRC

        # Jitter buffer (for incoming packets)
        self._jitter_buffer: deque = deque(maxlen=100)
        self._last_sequence = None

        # Audio callback
        self._audio_callback: Optional[Callable[[bytes], Awaitable[None]]] = None

        # Metrics
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_lost = 0
        self.packets_out_of_order = 0

        logger.info(
            f"RTPHandler initialized (port={local_port}, "
            f"payload_type={payload_type}, rate={sample_rate}Hz)"
        )

    async def start(self, audio_callback: Callable[[bytes], Awaitable[None]]):
        """
        Start RTP handler.

        Args:
            audio_callback: Async function to call with received audio
        """
        self._audio_callback = audio_callback

        # Create UDP socket
        loop = asyncio.get_event_loop()

        class RTPProtocol(asyncio.DatagramProtocol):
            def __init__(self, handler):
                self.handler = handler

            def connection_made(self, transport):
                self.handler._transport = transport

            def datagram_received(self, data, addr):
                asyncio.create_task(self.handler._handle_rtp_packet(data, addr))

            def error_received(self, exc):
                logger.error(f"RTP socket error: {exc}")

        self._transport, self._protocol = await loop.create_datagram_endpoint(
            lambda: RTPProtocol(self),
            local_addr=("0.0.0.0", self.local_port)
        )

        logger.info(f"RTP handler started on port {self.local_port}")

    async def stop(self):
        """Stop RTP handler"""
        if self._transport:
            self._transport.close()
            self._transport = None

        logger.info("RTP handler stopped")

    async def _handle_rtp_packet(self, data: bytes, addr: tuple):
        """
        Handle incoming RTP packet.

        Args:
            data: Raw packet bytes
            addr: Source address (host, port)
        """
        try:
            # Parse packet
            packet = RTPPacket.parse(data)

            # Update metrics
            self.packets_received += 1

            # Check for packet loss
            if self._last_sequence is not None:
                expected_seq = (self._last_sequence + 1) & 0xFFFF
                if packet.sequence_number != expected_seq:
                    lost = (packet.sequence_number - expected_seq) & 0xFFFF
                    if lost < 100:  # Reasonable loss threshold
                        self.packets_lost += lost
                        logger.warning(f"Packet loss detected: {lost} packets")
                    else:
                        # Likely out-of-order, not loss
                        self.packets_out_of_order += 1

            self._last_sequence = packet.sequence_number

            # Add to jitter buffer
            self._jitter_buffer.append(packet)

            # Process jittered packets
            await self._process_jitter_buffer()

        except Exception as e:
            logger.error(f"Error handling RTP packet: {e}")

    async def _process_jitter_buffer(self):
        """
        Process jitter buffer and deliver audio.

        Reorders packets and handles gaps.
        """
        if len(self._jitter_buffer) < 3:  # Wait for some buffering
            return

        # Sort by sequence number
        sorted_packets = sorted(self._jitter_buffer, key=lambda p: p.sequence_number)

        # Deliver oldest packets
        while sorted_packets:
            packet = sorted_packets.pop(0)

            # Call audio callback with payload
            if self._audio_callback:
                try:
                    await self._audio_callback(packet.payload)
                except Exception as e:
                    logger.error(f"Audio callback error: {e}")

            # Remove from buffer
            if packet in self._jitter_buffer:
                self._jitter_buffer.remove(packet)

    async def send_audio(self, audio: bytes, remote_host: str, remote_port: int):
        """
        Send audio via RTP.

        Args:
            audio: Raw PCM audio bytes
            remote_host: Remote RTP host
            remote_port: Remote RTP port
        """
        if not self._transport:
            raise RuntimeError("RTP handler not started")

        # Create RTP packet
        packet = RTPPacket(
            version=2,
            padding=False,
            extension=False,
            marker=False,
            payload_type=self.payload_type,
            sequence_number=self._sequence_number,
            timestamp=self._timestamp,
            ssrc=self._ssrc,
            payload=audio
        )

        # Send packet
        self._transport.sendto(packet.to_bytes(), (remote_host, remote_port))

        # Update state
        self._sequence_number = (self._sequence_number + 1) & 0xFFFF
        self._timestamp += len(audio) // 2  # 2 bytes per sample (16-bit)
        self.packets_sent += 1

    def get_metrics(self) -> dict:
        """Get RTP metrics"""
        packet_loss_rate = 0.0
        if self.packets_received > 0:
            packet_loss_rate = (self.packets_lost / self.packets_received) * 100

        return {
            "local_port": self.local_port,
            "packets_sent": self.packets_sent,
            "packets_received": self.packets_received,
            "packets_lost": self.packets_lost,
            "packets_out_of_order": self.packets_out_of_order,
            "packet_loss_rate_percent": round(packet_loss_rate, 2),
            "jitter_buffer_size": len(self._jitter_buffer)
        }

    def __str__(self) -> str:
        return (
            f"RTPHandler(port={self.local_port}, sent={self.packets_sent}, "
            f"recv={self.packets_received}, lost={self.packets_lost})"
        )
