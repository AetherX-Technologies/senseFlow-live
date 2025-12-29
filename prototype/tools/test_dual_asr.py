import os
import sys
import wave
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "prototype"))

from asr_engine import ASREngine, ASREvent, EventType
from asr_engine.engine import ASRConfig


def read_wav_mono(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sr


def resample(audio, src_rate, dst_rate):
    if src_rate == dst_rate or audio.size == 0:
        return audio
    duration = audio.shape[0] / float(src_rate)
    target_len = max(1, int(round(duration * dst_rate)))
    x_old = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32, copy=False)


def build_engine(model_path):
    print("init engine", model_path)
    cfg = ASRConfig(
        use_vad=True,
        use_punc=True,
        device="cuda:0",
        final_decode_model=model_path,
    )
    engine = ASREngine(cfg)
    if not engine.initialize():
        raise RuntimeError("ASR engine init failed")
    return engine


def main():
    root = ROOT
    print("root", root)
    mic_wav = os.path.join(
        root,
        "models",
        "models",
        "damo",
        "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        "example",
        "asr_example.wav",
    )
    sys_wav = os.path.join(root, "runtime", "triton_gpu", "client", "test_wavs", "mid.wav")
    model_path = os.path.join(
        root,
        "models",
        "models",
        "damo",
        "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    )

    print("load wavs")
    mic_audio, mic_sr = read_wav_mono(mic_wav)
    sys_audio, sys_sr = read_wav_mono(sys_wav)
    mic_audio = resample(mic_audio, mic_sr, 16000)
    sys_audio = resample(sys_audio, sys_sr, 16000)

    print("build engines")
    mic_engine = build_engine(model_path)
    sys_engine = build_engine(model_path)

    def on_event(source):
        def handler(event: ASREvent):
            if event.type == EventType.ASR_FINAL:
                text = (event.payload.get("text", "") or "").strip()
                if text:
                    print(f"[{source}] {text}")
        return handler

    for ev_type in EventType:
        mic_engine.on(ev_type, on_event("MIC"))
        sys_engine.on(ev_type, on_event("SYS"))

    chunk = 3200
    total = max(mic_audio.shape[0], sys_audio.shape[0])
    print("feed audio", total)
    for i in range(0, total, chunk):
        if i < mic_audio.shape[0]:
            mic_engine.feed_audio(mic_audio[i:i + chunk])
        if i < sys_audio.shape[0]:
            sys_engine.feed_audio(sys_audio[i:i + chunk])

    print("finalize")
    mic_engine.finalize()
    sys_engine.finalize()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("error", repr(exc))
        raise
