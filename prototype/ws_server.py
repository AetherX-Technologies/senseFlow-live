# -*- coding: utf-8 -*-
"""
WebSocket Server for ASR Engine with LLM Integration
Bridges ASREngine with frontend UI via WebSocket
"""

import asyncio
import json
import os
import sys
import ctypes
import queue
import numpy as np
import sounddevice as sd
from typing import Set

# Fix Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except:
        pass

# WebSocket
try:
    import websockets
    from websockets.server import serve
except ImportError:
    print("Installing websockets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets", "-q"])
    import websockets
    from websockets.server import serve

sys.path.insert(0, 'e:/code/FunASR-main/FunASR-main/prototype')

from asr_engine import ASREngine, ASREvent, EventType
from asr_engine.engine import ASRConfig
from asr_engine.llm_client import LLMClient, LLMConfig, InsightGenerator

# ANSI colors
GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
GRAY = '\033[90m'
RESET = '\033[0m'

# Audio settings
SAMPLE_RATE = 16000
BLOCK_SIZE = 3200  # 200ms


class ASRWebSocketServer:
    """WebSocket server that streams ASR events to connected clients"""

    def __init__(self, host: str = "localhost", port: int = 8766):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.engine: ASREngine = None
        self.audio_queue = queue.Queue()
        self.running = False
        self._loop = None

        # LLM integration
        base_dir = os.path.abspath(os.path.dirname(__file__))
        schema_path = os.path.join(base_dir, "tools", "llm_schema.json")
        llm_config = LLMConfig(
            base_url="http://127.0.0.1:8040/v1",
            api_key="46831818513nn!K",
            model="claude-haiku-4-5-20251001",
            use_claude_cli=True,
            cli_schema_path=schema_path
        )
        self.llm = LLMClient(llm_config)
        self.insights = InsightGenerator(self.llm)
        self._insight_task = None

    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        if not self.clients:
            return
        msg_str = json.dumps(message, ensure_ascii=False)
        await asyncio.gather(
            *[client.send(msg_str) for client in self.clients],
            return_exceptions=True
        )

    def _on_event(self, event: ASREvent):
        """Handle ASR event and queue for broadcast"""
        if self._loop:
            self._loop.call_soon_threadsafe(
                lambda e=event: asyncio.create_task(self._handle_event(e))
            )

    async def _handle_event(self, event: ASREvent):
        """Process ASR event and broadcast"""
        # Broadcast the event
        await self.broadcast(event.to_dict())

        # On final text, add to transcript and maybe generate insights
        if event.type == EventType.ASR_FINAL:
            text = event.payload.get("text", "")
            if text:
                self.insights.add_text(text)
                print(f"{CYAN}[LLM]{RESET} Added {len(text)} chars, total: {len(self.insights.full_transcript)}")

                # Check if we should generate insights
                if self.insights.should_generate_summary():
                    asyncio.create_task(self._generate_and_send_insights())

    async def _generate_and_send_insights(self):
        """Generate insights and send to clients"""
        try:
            print(f"{CYAN}[LLM]{RESET} Generating insights...")
            insights = await self.insights.generate_insights()

            summary_live = insights.get("summary_live", []) if insights else []
            if insights and (insights.get("summary") or summary_live or insights.get("actions") or insights.get("questions")):
                await self.broadcast({
                    "type": "insights.update",
                    "payload": insights
                })
                print(
                    f"{GREEN}[LLM]{RESET} Sent insights: {len(insights.get('summary', []))} summaries, "
                    f"{len(summary_live)} live summaries, {len(insights.get('actions', []))} actions, "
                    f"{len(insights.get('questions', []))} questions"
                )
            else:
                print(f"{GRAY}[LLM]{RESET} No insights generated (empty result)")
        except Exception as e:
            import traceback
            print(f"{YELLOW}[LLM]{RESET} Insight error ({type(e).__name__}): {e}")
            traceback.print_exc()

    def _audio_callback(self, indata, frames, time_info, status):
        """Audio capture callback"""
        if status:
            print(f"{GRAY}Audio: {status}{RESET}", file=sys.stderr)
        self.audio_queue.put(indata[:, 0].copy())

    async def _process_audio(self):
        """Process audio in async loop"""
        while self.running:
            try:
                # Non-blocking check for audio
                try:
                    audio_chunk = self.audio_queue.get_nowait()
                    self.engine.feed_audio(audio_chunk)
                except queue.Empty:
                    pass
                await asyncio.sleep(0.01)  # Small delay to prevent CPU spin
            except Exception as e:
                print(f"{YELLOW}[Audio] Error: {e}{RESET}")

    async def handler(self, websocket: websockets.WebSocketServerProtocol):
        """Handle WebSocket connection"""
        self.clients.add(websocket)
        client_id = id(websocket)
        print(f"{GREEN}[WS]{RESET} Client connected: {client_id}")

        # Send welcome message
        await websocket.send(json.dumps({
            "type": "connection.established",
            "payload": {
                "session_id": self.engine.session_id,
                "message": "Connected to ASR WebSocket Server with LLM"
            }
        }, ensure_ascii=False))

        try:
            async for message in websocket:
                # Handle client messages
                try:
                    data = json.loads(message)
                    cmd = data.get("command")

                    if cmd == "reset":
                        self.engine.reset()
                        self.insights.reset()
                        await websocket.send(json.dumps({
                            "type": "engine.reset",
                            "payload": {"status": "ok"}
                        }))

                    elif cmd == "ask":
                        # Handle Q&A
                        question = data.get("question", "")
                        if question:
                            print(f"{CYAN}[QA]{RESET} Question: {question}")
                            answer = await self.insights.answer(question)
                            await websocket.send(json.dumps({
                                "type": "qa.answer",
                                "payload": {
                                    "question": question,
                                    "answer": answer
                                }
                            }, ensure_ascii=False))
                            print(f"{GREEN}[QA]{RESET} Answered: {answer[:50]}...")

                    elif cmd == "generate_insights":
                        # Force generate insights
                        asyncio.create_task(self._generate_and_send_insights())

                except json.JSONDecodeError:
                    pass
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"{GRAY}[WS]{RESET} Client disconnected: {client_id}")

    async def start(self):
        """Start the WebSocket server"""
        print("=" * 60)
        print(f"{GREEN}ASR WebSocket Server + LLM{RESET}")
        print("=" * 60)

        # Initialize ASR Engine
        print("\n[1/3] Initializing ASR Engine...")
        config = ASRConfig(
            use_vad=True,
            use_punc=True,
            device="cuda:0",
            final_decode_model=(
                "E:/code/FunASR-main/FunASR-main/models/models/damo/"
                "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
            ),
        )
        self.engine = ASREngine(config)

        # Register event handlers
        for event_type in EventType:
            self.engine.on(event_type, self._on_event)

        if not self.engine.initialize():
            print("Failed to initialize ASR Engine")
            return

        # Test LLM connection
        print(f"\n[2/3] Testing LLM connection...")
        print(f"  Model: {self.llm.config.model}")
        print(f"  URL: {self.llm.config.base_url}")

        # Start audio stream
        print(f"\n[3/3] Starting audio capture...")
        default_input = sd.default.device[0]
        device_info = sd.query_devices(default_input)
        print(f"  Microphone: {device_info['name']}")

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=BLOCK_SIZE,
            callback=self._audio_callback
        )

        self.running = True
        self._loop = asyncio.get_running_loop()

        print(f"\n{'-' * 60}")
        print(f"WebSocket: ws://{self.host}:{self.port}")
        print(f"Open prototype/index_live.html in browser")
        print(f"Press {GREEN}Ctrl+C{RESET} to stop")
        print(f"{'-' * 60}\n")

        # Start WebSocket server
        async with serve(self.handler, self.host, self.port):
            with stream:
                # Run audio processing
                await self._process_audio()

    async def cleanup(self):
        """Cleanup resources"""
        await self.llm.close()

    def stop(self):
        """Stop the server"""
        self.running = False
        if self.engine:
            self.engine.finalize()


async def main():
    server = ASRWebSocketServer(host="localhost", port=8766)
    try:
        await server.start()
    except KeyboardInterrupt:
        print(f"\n{GREEN}Shutting down...{RESET}")
        server.stop()
        await server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
