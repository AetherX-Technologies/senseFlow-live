# -*- coding: utf-8 -*-
"""
FunASR streaming ASR model verification
"""

from funasr import AutoModel
import soundfile
import os
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except Exception:
        pass
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Streaming parameters
chunk_size = [0, 10, 5]          # 600ms realtime step, 300ms future context
encoder_chunk_look_back = 4
decoder_chunk_look_back = 1

print("=" * 50)
print("FunASR Streaming ASR Verification")
print("=" * 50)

# Model path (ASR only for streaming test)
model_base = "E:/code/FunASR-main/FunASR-main/models/models/damo"
asr_model = f"{model_base}/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"

print(f"\n[1/4] Loading ASR model...")
print(f"  Model: {os.path.basename(asr_model)}")

model = AutoModel(
    model=asr_model,
    device="cuda:0",
    disable_update=True,
)
print("  [OK] Model loaded")

# Load test audio
wav_file = os.path.join(model.model_path, "example/asr_example.wav")
print(f"\n[2/4] Loading audio: {os.path.basename(wav_file)}")
speech, sample_rate = soundfile.read(wav_file)
print(f"  Sample rate: {sample_rate} Hz")
print(f"  Duration: {len(speech)/sample_rate:.2f} sec")
print(f"  Samples: {len(speech)}")

# Streaming inference
chunk_stride = chunk_size[1] * 960  # 600ms @ 16kHz
total_chunk_num = int((len(speech) - 1) / chunk_stride + 1)

print(f"\n[3/4] Streaming ({total_chunk_num} chunks, {chunk_stride/sample_rate*1000:.0f}ms each)")
print("-" * 50)

cache = {}
all_text = []

for i in range(total_chunk_num):
    chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
    is_final = i == total_chunk_num - 1

    res = model.generate(
        input=chunk,
        cache=cache,
        is_final=is_final,
        chunk_size=chunk_size,
        encoder_chunk_look_back=encoder_chunk_look_back,
        decoder_chunk_look_back=decoder_chunk_look_back,
    )

    # Extract text
    if res and len(res) > 0 and 'text' in res[0]:
        text = res[0]['text']
        if text:
            all_text.append(text)
            status = "FINAL" if is_final else "DRAFT"
            print(f"  [{i+1}/{total_chunk_num}] [{status}] {text}")

print("-" * 50)
print(f"\n[4/4] Full transcription:")
print(f"  {''.join(all_text)}")
print("\n" + "=" * 50)
print("[OK] Model verification complete")
print("=" * 50)
