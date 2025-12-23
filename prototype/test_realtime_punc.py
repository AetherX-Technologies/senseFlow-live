# -*- coding: utf-8 -*-
"""
FunASR Realtime ASR with Punctuation
Real-time speech recognition with live display and punctuation restoration
"""

from funasr import AutoModel
import sounddevice as sd
import numpy as np
import sys
import queue
import ctypes

# Fix Windows console encoding and enable ANSI colors
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    # Enable ANSI escape sequences on Windows 10+
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

# ANSI colors for terminal
GRAY = '\033[90m'
WHITE = '\033[97m'
GREEN = '\033[92m'
RESET = '\033[0m'
CLEAR_LINE = '\033[2K\r'

# Streaming parameters
SAMPLE_RATE = 16000
CHUNK_SIZE = [0, 10, 5]  # 600ms chunks
ENCODER_CHUNK_LOOK_BACK = 4
DECODER_CHUNK_LOOK_BACK = 1
CHUNK_STRIDE = CHUNK_SIZE[1] * 960  # 9600 samples = 600ms

# Audio buffer
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Callback for sounddevice to capture audio"""
    if status:
        print(f"{GRAY}Audio: {status}{RESET}", file=sys.stderr)
    audio_queue.put(indata[:, 0].copy())

def main():
    print("=" * 70)
    print(f"{GREEN}FunASR Realtime ASR + Punctuation{RESET}")
    print("=" * 70)

    # Model paths (local)
    model_base = "E:/code/FunASR-main/FunASR-main/models/models/damo"
    asr_model_path = f"{model_base}/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
    punc_model_path = f"{model_base}/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

    # Load ASR model
    print(f"\n[1/3] Loading ASR model...")
    asr_model = AutoModel(
        model=asr_model_path,
        device="cuda:0",
        disable_update=True,
    )
    print(f"  {GREEN}[OK]{RESET} ASR model loaded")

    # Load Punctuation model
    print(f"\n[2/3] Loading Punctuation model...")
    try:
        punc_model = AutoModel(
            model=punc_model_path,
            device="cuda:0",
            disable_update=True,
        )
        print(f"  {GREEN}[OK]{RESET} Punctuation model loaded")
        use_punc = True
    except Exception as e:
        print(f"  {GRAY}[SKIP] Punctuation model not available: {e}{RESET}")
        use_punc = False

    # Audio device info
    print(f"\n[3/3] Starting realtime recognition...")
    default_input = sd.default.device[0]
    device_info = sd.query_devices(default_input)
    print(f"  Microphone: {device_info['name']}")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Chunk: {CHUNK_STRIDE/SAMPLE_RATE*1000:.0f}ms")
    print("-" * 70)
    print(f"Speak now! Press {GREEN}Ctrl+C{RESET} to stop.")
    print("-" * 70)
    print()

    # Start audio stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=CHUNK_STRIDE,
        callback=audio_callback
    )

    asr_cache = {}
    punc_cache = {}
    audio_buffer = np.array([], dtype=np.float32)

    # Text accumulation
    all_text_parts = []  # Store all finalized text parts
    current_draft = ""   # Current draft text (not finalized)
    display_text = ""    # Full display text

    chunk_count = 0

    try:
        with stream:
            while True:
                try:
                    audio_chunk = audio_queue.get(timeout=0.1)
                    audio_buffer = np.concatenate([audio_buffer, audio_chunk])

                    while len(audio_buffer) >= CHUNK_STRIDE:
                        chunk = audio_buffer[:CHUNK_STRIDE]
                        audio_buffer = audio_buffer[CHUNK_STRIDE:]
                        chunk_count += 1

                        # ASR inference
                        res = asr_model.generate(
                            input=chunk,
                            cache=asr_cache,
                            is_final=False,
                            chunk_size=CHUNK_SIZE,
                            encoder_chunk_look_back=ENCODER_CHUNK_LOOK_BACK,
                            decoder_chunk_look_back=DECODER_CHUNK_LOOK_BACK,
                            disable_pbar=True,
                        )

                        # Extract text
                        if res and len(res) > 0 and 'text' in res[0]:
                            text = res[0]['text']
                            if text:
                                current_draft = text

                                # Build display: finalized + current draft
                                finalized = ''.join(all_text_parts)

                                # Apply punctuation to accumulated text
                                if use_punc and len(finalized) > 0:
                                    try:
                                        punc_res = punc_model.generate(
                                            input=finalized + current_draft,
                                            cache=punc_cache,
                                            disable_pbar=True,
                                        )
                                        if punc_res and len(punc_res) > 0:
                                            display_text = punc_res[0]['text']
                                    except:
                                        display_text = finalized + current_draft
                                else:
                                    display_text = finalized + f"{GRAY}{current_draft}{RESET}"

                                # Real-time display (update current line)
                                # Show last 60 chars to fit terminal
                                show_text = display_text[-60:] if len(display_text) > 60 else display_text
                                print(f"{CLEAR_LINE}{GREEN}>{RESET} {show_text}", end='', flush=True)

                                # Accumulate text periodically (every 5 chunks with text)
                                if chunk_count % 5 == 0 and current_draft:
                                    all_text_parts.append(current_draft)
                                    current_draft = ""

                except queue.Empty:
                    continue

    except KeyboardInterrupt:
        print(f"\n\n{'-' * 70}")
        print(f"{GREEN}Recording stopped.{RESET}")

        # Final processing
        if len(audio_buffer) > 0:
            res = asr_model.generate(
                input=audio_buffer,
                cache=asr_cache,
                is_final=True,
                chunk_size=CHUNK_SIZE,
                encoder_chunk_look_back=ENCODER_CHUNK_LOOK_BACK,
                decoder_chunk_look_back=DECODER_CHUNK_LOOK_BACK,
                disable_pbar=True,
            )
            if res and len(res) > 0 and 'text' in res[0]:
                final_text = res[0]['text']
                if final_text:
                    all_text_parts.append(final_text)

        # Add remaining draft
        if current_draft:
            all_text_parts.append(current_draft)

        # Final transcription
        full_text = ''.join(all_text_parts)

        # Apply punctuation to final result
        if use_punc and full_text:
            try:
                punc_res = punc_model.generate(
                    input=full_text,
                    cache={},  # Fresh cache for final
                    disable_pbar=True,
                )
                if punc_res and len(punc_res) > 0:
                    full_text = punc_res[0]['text']
            except Exception as e:
                print(f"{GRAY}Punctuation error: {e}{RESET}")

        print(f"\n{'=' * 70}")
        print(f"{GREEN}Full Transcription:{RESET}")
        print(f"\n{full_text}\n")
        print(f"{'=' * 70}")
        print(f"Total chunks: {chunk_count}")
        print(f"{GREEN}[OK]{RESET} Complete")

if __name__ == "__main__":
    main()
