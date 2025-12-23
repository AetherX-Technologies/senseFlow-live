# -*- coding: utf-8 -*-
"""
ASR Engine Integration Test
Real-time speech recognition with VAD-gated ASR and punctuation
"""

import sys
import ctypes
import queue
import numpy as np
import sounddevice as sd

# Fix Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    kernel32 = ctypes.windll.kernel32
    kernel32.SetStdHandle(-11, 7)
    try:
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except:
        pass

# Add parent to path for imports
sys.path.insert(0, 'e:/code/FunASR-main/FunASR-main/prototype')

from asr_engine import ASREngine, ASREvent, EventType
from asr_engine.engine import ASRConfig

# ANSI colors
GRAY = '\033[90m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'
CLEAR_LINE = '\033[2K\r'

# Audio settings
SAMPLE_RATE = 16000
BLOCK_SIZE = 3200  # 200ms chunks to match VAD


def main():
    print("=" * 70)
    print(f"{GREEN}ASR Engine Integration Test{RESET}")
    print("=" * 70)

    # Audio queue
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"{GRAY}Audio: {status}{RESET}", file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    # Event handlers
    current_text = [""]  # Use list for closure

    def on_ready(event: ASREvent):
        print(f"\n{GREEN}[ENGINE]{RESET} Ready")
        print(f"  Session: {event.session_id}")
        print(f"  VAD: {event.payload.get('vad_enabled')}")
        print(f"  Punc: {event.payload.get('punc_enabled')}")

    def on_speech_start(event: ASREvent):
        print(f"\n{CYAN}[VAD]{RESET} Speech started (segment: {event.segment_id})")
        current_text[0] = ""

    def on_speech_end(event: ASREvent):
        print(f"\n{CYAN}[VAD]{RESET} Speech ended (segment: {event.segment_id})")

    def on_partial(event: ASREvent):
        text = event.payload.get('text', '')
        current_text[0] = text
        # Show last 60 chars
        display = text[-60:] if len(text) > 60 else text
        print(f"{CLEAR_LINE}{GRAY}[DRAFT]{RESET} {display}", end='', flush=True)

    def on_final(event: ASREvent):
        text = event.payload.get('text', '')
        duration = event.payload.get('duration_ms', 0)
        print(f"\n{GREEN}[FINAL]{RESET} {text}")
        print(f"{GRAY}        Duration: {duration}ms | Segment: {event.segment_id}{RESET}")

    def on_error(event: ASREvent):
        print(f"\n{YELLOW}[ERROR]{RESET} {event.payload.get('error')}")

    # Create engine
    config = ASRConfig(
        use_vad=True,
        use_punc=True,
        device="cuda:0",
        final_decode_model=(
            "E:/code/FunASR-main/FunASR-main/models/models/damo/"
            "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        ),
    )
    engine = ASREngine(config)

    # Register event handlers
    engine.on(EventType.ENGINE_READY, on_ready)
    engine.on(EventType.VAD_SPEECH_START, on_speech_start)
    engine.on(EventType.VAD_SPEECH_END, on_speech_end)
    engine.on(EventType.ASR_PARTIAL, on_partial)
    engine.on(EventType.ASR_FINAL, on_final)
    engine.on(EventType.ENGINE_ERROR, on_error)

    # Initialize
    print("\nInitializing ASR Engine...")
    if not engine.initialize():
        print("Failed to initialize engine")
        return

    # Audio device info
    print(f"\n{'-' * 70}")
    default_input = sd.default.device[0]
    device_info = sd.query_devices(default_input)
    print(f"Microphone: {device_info['name']}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Block size: {BLOCK_SIZE} samples ({BLOCK_SIZE/SAMPLE_RATE*1000:.0f}ms)")
    print(f"{'-' * 70}")
    print(f"Speak now! Press {GREEN}Ctrl+C{RESET} to stop.")
    print(f"VAD will detect speech and trigger ASR automatically.")
    print(f"{'-' * 70}\n")

    # Start audio stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=BLOCK_SIZE,
        callback=audio_callback
    )

    try:
        with stream:
            while True:
                try:
                    audio_chunk = audio_queue.get(timeout=0.1)
                    engine.feed_audio(audio_chunk)
                except queue.Empty:
                    continue

    except KeyboardInterrupt:
        print(f"\n\n{'-' * 70}")
        print(f"{GREEN}Stopping...{RESET}")

        # Finalize
        engine.finalize()

        print(f"\n{GREEN}[OK]{RESET} Session complete")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
