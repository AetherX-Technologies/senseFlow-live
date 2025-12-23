# -*- coding: utf-8 -*-
"""
FunASR Microphone Realtime Streaming ASR Test
Real-time speech recognition using microphone input
"""

from funasr import AutoModel
import sounddevice as sd
import numpy as np
import sys
import threading
import queue

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Streaming parameters
SAMPLE_RATE = 16000
CHUNK_SIZE = [0, 10, 5]  # 600ms chunks
ENCODER_CHUNK_LOOK_BACK = 4
DECODER_CHUNK_LOOK_BACK = 1
CHUNK_STRIDE = CHUNK_SIZE[1] * 960  # samples per chunk (600ms @ 16kHz = 9600 samples)

# Audio buffer
audio_queue = queue.Queue()
is_running = True

def audio_callback(indata, frames, time, status):
    """Callback for sounddevice to capture audio"""
    if status:
        print(f"Audio status: {status}", file=sys.stderr)
    # Convert to mono float32 and add to queue
    audio_queue.put(indata[:, 0].copy())

def main():
    global is_running

    print("=" * 60)
    print("FunASR Microphone Realtime ASR Test")
    print("=" * 60)

    # Model path (local)
    model_base = "E:/code/FunASR-main/FunASR-main/models/models/damo"
    asr_model = f"{model_base}/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"

    print(f"\n[1/3] Loading ASR model...")
    print(f"  Model: speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online")

    model = AutoModel(
        model=asr_model,
        device="cuda:0",
        disable_update=True,
    )
    print("  [OK] Model loaded")

    # List available audio devices
    print(f"\n[2/3] Audio devices:")
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            marker = " <-- DEFAULT" if i == default_input else ""
            print(f"  [{i}] {d['name']}{marker}")

    print(f"\n[3/3] Starting realtime recognition...")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Chunk size: {CHUNK_STRIDE} samples ({CHUNK_STRIDE/SAMPLE_RATE*1000:.0f}ms)")
    print("-" * 60)
    print("Speak into your microphone. Press Ctrl+C to stop.")
    print("-" * 60)

    # Start audio stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=CHUNK_STRIDE,
        callback=audio_callback
    )

    cache = {}
    audio_buffer = np.array([], dtype=np.float32)
    chunk_count = 0
    all_text = []

    try:
        with stream:
            while is_running:
                try:
                    # Get audio from queue (with timeout to allow Ctrl+C)
                    audio_chunk = audio_queue.get(timeout=0.1)
                    audio_buffer = np.concatenate([audio_buffer, audio_chunk])

                    # Process when we have enough samples
                    while len(audio_buffer) >= CHUNK_STRIDE:
                        chunk = audio_buffer[:CHUNK_STRIDE]
                        audio_buffer = audio_buffer[CHUNK_STRIDE:]
                        chunk_count += 1

                        # Run ASR inference
                        res = model.generate(
                            input=chunk,
                            cache=cache,
                            is_final=False,
                            chunk_size=CHUNK_SIZE,
                            encoder_chunk_look_back=ENCODER_CHUNK_LOOK_BACK,
                            decoder_chunk_look_back=DECODER_CHUNK_LOOK_BACK,
                        )

                        # Extract and display text
                        if res and len(res) > 0 and 'text' in res[0]:
                            text = res[0]['text']
                            if text:
                                all_text.append(text)
                                print(f"  [Chunk {chunk_count}] {text}")

                except queue.Empty:
                    continue

    except KeyboardInterrupt:
        print("\n" + "-" * 60)
        print("Stopping...")
        is_running = False

        # Final inference
        if len(audio_buffer) > 0:
            res = model.generate(
                input=audio_buffer,
                cache=cache,
                is_final=True,
                chunk_size=CHUNK_SIZE,
                encoder_chunk_look_back=ENCODER_CHUNK_LOOK_BACK,
                decoder_chunk_look_back=DECODER_CHUNK_LOOK_BACK,
            )
            if res and len(res) > 0 and 'text' in res[0]:
                text = res[0]['text']
                if text:
                    all_text.append(text)
                    print(f"  [FINAL] {text}")

    print("\n" + "=" * 60)
    print("Full transcription:")
    print(f"  {''.join(all_text)}")
    print("=" * 60)
    print(f"Total chunks processed: {chunk_count}")
    print("[OK] Test complete")

if __name__ == "__main__":
    main()
