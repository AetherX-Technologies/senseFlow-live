# -*- coding: utf-8 -*-
"""
ASREngine - Main streaming ASR engine with VAD and punctuation

Architecture: ASR runs continuously, VAD only marks segment boundaries.
This prevents word dropping at speech start/end.

Pipeline:
  Audio -> ASR (always running) -> Partial events
       \-> VAD (parallel) -> Segment markers -> Final events with punctuation
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any, Tuple
import queue
import threading
import os
import numpy as np
import uuid
import time

from funasr import AutoModel

from .events import EventEmitter, EventType, ASREvent
from .vad import VADProcessor, VADConfig, SpeechSegment


@dataclass
class ASRConfig:
    """ASR Engine configuration"""
    # Model paths
    model_base: str = "E:/code/FunASR-main/FunASR-main/models/models/damo"

    # Audio settings
    sample_rate: int = 16000

    # ASR streaming parameters (from architecture.md)
    asr_chunk_size: List[int] = None  # [0, 10, 5] -> 600ms
    encoder_chunk_look_back: int = 4
    decoder_chunk_look_back: int = 1

    # VAD settings
    vad_chunk_size_ms: int = 200
    use_vad: bool = True
    pre_speech_ms: int = 400
    vad_end_padding_ms: int = 800

    # Final re-decode settings (scheme C)
    final_decode_enabled: bool = True
    final_decode_device: Optional[str] = None  # None -> same as ASR device
    final_decode_model: Optional[str] = None  # None -> auto select offline if available
    final_decode_queue_size: int = 8
    final_decode_min_ms: int = 200  # skip very short segments

    # Punctuation settings
    use_punc: bool = True
    punc_on_partial: bool = False

    # Partial display behavior
    emit_partial_without_segment: bool = True
    live_partial_segment_id: str = "live"

    # Device
    device: str = "cuda:0"

    def __post_init__(self):
        if self.asr_chunk_size is None:
            self.asr_chunk_size = [0, 10, 5]

    @property
    def asr_chunk_stride(self) -> int:
        """ASR chunk stride in samples (600ms = 9600 samples)"""
        return self.asr_chunk_size[1] * 960

    @property
    def vad_chunk_stride(self) -> int:
        """VAD chunk stride in samples (200ms = 3200 samples)"""
        return int(self.vad_chunk_size_ms * self.sample_rate / 1000)

    @property
    def pre_speech_samples(self) -> int:
        """Pre-roll buffer size in samples"""
        return int(self.pre_speech_ms * self.sample_rate / 1000)

    @property
    def vad_end_padding_samples(self) -> int:
        """Extra tail padding after VAD end in samples"""
        return int(self.vad_end_padding_ms * self.sample_rate / 1000)


class ASREngine:
    """
    Streaming ASR Engine with VAD and punctuation

    Events emitted:
    - engine.ready: Engine initialized
    - vad.speech.start: Speech detected
    - vad.speech.end: Speech ended
    - asr.partial: Draft transcription
    - asr.final: Final transcription with punctuation
    - engine.error: Error occurred
    """

    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        self.events = EventEmitter()

        # Models (loaded on init)
        self._asr_model = None
        self._vad_model = None
        self._punc_model = None
        self._vad_processor: Optional[VADProcessor] = None
        self._final_asr_model = None
        self._final_is_streaming = True

        # State
        self._asr_cache = {}
        self._punc_cache = {}
        self._is_initialized = False
        self._punc_lock = threading.Lock()
        self._final_queue: "queue.Queue[Optional[Tuple[str, np.ndarray, float, float, str]]]" = queue.Queue(
            maxsize=self.config.final_decode_queue_size
        )
        self._final_thread: Optional[threading.Thread] = None
        self._final_running = False
        self._silence_samples = 0

        # Buffers
        self._vad_buffer = np.array([], dtype=np.float32)
        self._asr_buffer = np.array([], dtype=np.float32)
        self._pending_end_samples = 0
        self._pre_audio_buffer = np.array([], dtype=np.float32)
        self._segment_audio = np.array([], dtype=np.float32)

        # Segment tracking
        self._current_segment_id: Optional[str] = None
        self._segment_text_parts: List[str] = []
        self._segment_start_time: float = 0

        # Pending text buffer (for lookback on VAD start)
        # Stores recent ASR output before VAD detects speech
        self._pending_text_parts: List[str] = []
        self._max_pending_chars = 100  # Keep last ~100 chars for lookback
        self._live_segment_id = self.config.live_partial_segment_id

    # === Event API ===

    def on(self, event_type: EventType, callback: Callable[[ASREvent], None]):
        """Register event listener"""
        self.events.on(event_type, callback)

    def off(self, event_type: EventType, callback: Callable[[ASREvent], None]):
        """Remove event listener"""
        self.events.off(event_type, callback)

    # === Lifecycle ===

    def initialize(self) -> bool:
        """Load models and initialize engine"""
        try:
            model_base = self.config.model_base

            # Load ASR model
            print("[ASREngine] Loading ASR model...")
            self._asr_model = AutoModel(
                model=f"{model_base}/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
                device=self.config.device,
                disable_update=True,
            )
            if self.config.final_decode_enabled:
                final_device = self.config.final_decode_device or self.config.device
                try:
                    final_model = self._resolve_final_model(model_base)
                    print("[ASREngine] Loading Final ASR model...")
                    self._final_asr_model = AutoModel(
                        model=final_model,
                        device=final_device,
                        disable_update=True,
                    )
                    self._final_is_streaming = "online" in str(final_model)
                except Exception as e:
                    print(f"[ASREngine] Final ASR model load failed: {e}")
                    print("[ASREngine] Falling back to streaming-only final")
                    self._final_asr_model = None
                    self.config.final_decode_enabled = False

            # Load VAD model
            if self.config.use_vad:
                print("[ASREngine] Loading VAD model...")
                self._vad_model = AutoModel(
                    model=f"{model_base}/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                    device=self.config.device,
                    disable_update=True,
                )
                vad_config = VADConfig(
                    chunk_size_ms=self.config.vad_chunk_size_ms,
                    sample_rate=self.config.sample_rate
                )
                self._vad_processor = VADProcessor(self._vad_model, vad_config)

            # Load Punctuation model
            if self.config.use_punc:
                print("[ASREngine] Loading Punctuation model...")
                self._punc_model = AutoModel(
                    model=f"{model_base}/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                    device=self.config.device,
                    disable_update=True,
                )

            self._is_initialized = True
            self.events.emit(EventType.ENGINE_READY, {
                "vad_enabled": self.config.use_vad,
                "punc_enabled": self.config.use_punc,
            })
            if self.config.final_decode_enabled:
                self._start_final_worker()
            print("[ASREngine] Ready")
            return True

        except Exception as e:
            self.events.emit(EventType.ENGINE_ERROR, {"error": str(e)})
            print(f"[ASREngine] Initialization error: {e}")
            return False

    def reset(self):
        """Reset engine state for new session"""
        self._asr_cache = {}
        self._punc_cache = {}
        self._vad_buffer = np.array([], dtype=np.float32)
        self._asr_buffer = np.array([], dtype=np.float32)
        self._pending_end_samples = 0
        self._current_segment_id = None
        self._segment_text_parts = []
        self._pending_text_parts = []
        self._pre_audio_buffer = np.array([], dtype=np.float32)
        self._segment_audio = np.array([], dtype=np.float32)
        self._silence_samples = 0
        if self._vad_processor:
            self._vad_processor.reset()

    # === Processing ===

    def feed_audio(self, audio_chunk: np.ndarray):
        """
        Feed audio data to the engine

        Args:
            audio_chunk: Float32 audio samples at 16kHz
        """
        if not self._is_initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        if self.config.use_vad:
            self._process_with_vad(audio_chunk)
        else:
            self._process_without_vad(audio_chunk)

    def _process_with_vad(self, audio_chunk: np.ndarray):
        """
        Process audio with VAD for segmentation only (no gating).

        Architecture: ASR runs continuously, VAD only marks segment boundaries.
        This prevents word dropping at speech start/end.
        """
        # === 1. Always feed audio to ASR (no gating) ===
        self._asr_buffer = np.concatenate([self._asr_buffer, audio_chunk])
        self._process_asr_buffer()

        # === 2. VAD runs in parallel for segmentation only ===
        self._vad_buffer = np.concatenate([self._vad_buffer, audio_chunk])
        vad_stride = self.config.vad_chunk_stride

        while len(self._vad_buffer) >= vad_stride:
            vad_chunk = self._vad_buffer[:vad_stride]
            self._vad_buffer = self._vad_buffer[vad_stride:]

            # Run VAD
            is_speech, completed_segment = self._vad_processor.process_chunk(vad_chunk)

            # Handle speech start - emit event but don't gate ASR
            if (is_speech or completed_segment) and self._current_segment_id is None:
                self._start_segment()
                self._silence_samples = 0

            # Reset silence counter if speech resumes
            if is_speech:
                self._silence_samples = 0

            # Append audio to current segment (includes tail padding)
            if self._current_segment_id is not None:
                self._segment_audio = np.concatenate([self._segment_audio, vad_chunk])
            else:
                self._pre_audio_buffer = self._append_pre_audio(self._pre_audio_buffer, vad_chunk)

            # Handle speech end by silence duration (avoid micro-segments)
            if (not is_speech) and self._current_segment_id:
                self._silence_samples += vad_stride
                if self._silence_samples >= self.config.vad_end_padding_samples:
                    self._end_segment(completed_segment)
                    self._silence_samples = 0

    def _process_without_vad(self, audio_chunk: np.ndarray):
        """Process audio without VAD (always active)"""
        if self._current_segment_id is None:
            self._start_segment()

        self._asr_buffer = np.concatenate([self._asr_buffer, audio_chunk])
        self._process_asr_buffer()

    def _process_asr_buffer(self):
        """Process ASR buffer when enough data available"""
        asr_stride = self.config.asr_chunk_stride

        while len(self._asr_buffer) >= asr_stride:
            asr_chunk = self._asr_buffer[:asr_stride]
            self._asr_buffer = self._asr_buffer[asr_stride:]

            # Run ASR
            res = self._asr_model.generate(
                input=asr_chunk,
                cache=self._asr_cache,
                is_final=False,
                chunk_size=self.config.asr_chunk_size,
                encoder_chunk_look_back=self.config.encoder_chunk_look_back,
                decoder_chunk_look_back=self.config.decoder_chunk_look_back,
                disable_pbar=True,
            )

            # Extract and emit partial result
            if res and len(res) > 0 and 'text' in res[0]:
                text = res[0]['text']
                if text:
                    self._emit_partial(text)

    def _start_segment(self):
        """Start new speech segment with lookback from pending buffer"""
        self._current_segment_id = str(uuid.uuid4())[:8]
        self._segment_start_time = time.time()

        # Lookback: copy pending text to segment (captures pre-VAD speech)
        if self._pending_text_parts:
            self._segment_text_parts = self._pending_text_parts.copy()
            self._pending_text_parts = []
        else:
            self._segment_text_parts = []

        # Note: Keep ASR cache for continuity (don't reset)
        self._punc_cache = {}
        if len(self._pre_audio_buffer) > 0:
            self._segment_audio = self._pre_audio_buffer.copy()
            self._pre_audio_buffer = np.array([], dtype=np.float32)
        else:
            self._segment_audio = np.array([], dtype=np.float32)

        self.events.emit(
            EventType.VAD_SPEECH_START,
            {"start_time": self._segment_start_time},
            segment_id=self._current_segment_id
        )
        self.events.emit(
            EventType.AUDIO_SEGMENT_STARTED,
            {"segment_id": self._current_segment_id, "start_ts": self._segment_start_time},
            segment_id=self._current_segment_id
        )

        # Emit initial partial with lookback text if any
        if self._segment_text_parts:
            accumulated = ''.join(self._segment_text_parts)
            self.events.emit(
                EventType.ASR_PARTIAL,
                {"text": accumulated, "raw_text": accumulated, "confidence": 0.8},
                segment_id=self._current_segment_id
            )

    def _end_segment(self, vad_segment: Optional[SpeechSegment] = None):
        """End current speech segment and emit final result (async re-decode)"""
        if not self._current_segment_id:
            return

        end_time = time.time()
        segment_id = self._current_segment_id
        segment_audio = self._segment_audio
        fallback_text = ''.join(self._segment_text_parts)
        self._segment_audio = np.array([], dtype=np.float32)

        self.events.emit(
            EventType.VAD_SPEECH_END,
            {"end_time": end_time},
            segment_id=self._current_segment_id
        )
        self.events.emit(
            EventType.AUDIO_SEGMENT_ENDED,
            {"segment_id": self._current_segment_id, "end_ts": end_time},
            segment_id=self._current_segment_id
        )

        if self.config.final_decode_enabled:
            if len(segment_audio) >= int(self.config.final_decode_min_ms * self.config.sample_rate / 1000):
                try:
                    self._final_queue.put_nowait(
                        (segment_id, segment_audio, self._segment_start_time, end_time, fallback_text)
                    )
                except queue.Full:
                    self._emit_fallback_final(segment_id, self._segment_start_time, end_time, fallback_text)
            else:
                self._emit_fallback_final(segment_id, self._segment_start_time, end_time, fallback_text)
        else:
            self._emit_fallback_final(segment_id, self._segment_start_time, end_time, fallback_text)

        # Reset segment state
        self._current_segment_id = None
        self._segment_text_parts = []

    def _emit_partial(self, text: str):
        """Emit partial transcription, optionally without an active segment"""
        is_streaming = False
        segment_id = self._current_segment_id

        if segment_id is None:
            # Buffer for lookback
            self._pending_text_parts.append(text)
            pending_text = ''.join(self._pending_text_parts)
            if len(pending_text) > self._max_pending_chars:
                self._pending_text_parts = [pending_text[-self._max_pending_chars:]]

            if not self.config.emit_partial_without_segment:
                return

            is_streaming = True
            segment_id = self._live_segment_id
            accumulated = ''.join(self._pending_text_parts)
        else:
            # Active segment - emit partial
            self._segment_text_parts.append(text)
            accumulated = ''.join(self._segment_text_parts)

        # Optionally apply punctuation to partial
        display_text = accumulated
        if self.config.punc_on_partial and self.config.use_punc and self._punc_model and len(accumulated) > 5:
            try:
                with self._punc_lock:
                    punc_res = self._punc_model.generate(
                        input=accumulated,
                        cache=self._punc_cache,
                        disable_pbar=True,
                    )
                    if punc_res and len(punc_res) > 0 and 'text' in punc_res[0]:
                        display_text = punc_res[0]['text']
            except:
                pass

        self.events.emit(
            EventType.ASR_PARTIAL,
            {
                "text": display_text,
                "raw_text": accumulated,
                "confidence": 0.8,
                "streaming": is_streaming,
            },
            segment_id=segment_id
        )

    def _append_pre_audio(self, buffer: np.ndarray, chunk: np.ndarray) -> np.ndarray:
        """Append audio to pre-roll buffer with max length"""
        max_len = self.config.pre_speech_samples
        if max_len <= 0:
            return np.array([], dtype=np.float32)
        if buffer.size == 0:
            merged = chunk
        else:
            merged = np.concatenate([buffer, chunk])
        if len(merged) > max_len:
            merged = merged[-max_len:]
        return merged

    def _resolve_final_model(self, model_base: str) -> str:
        if self.config.final_decode_model:
            return self.config.final_decode_model
        offline_path = os.path.join(
            model_base,
            "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        )
        if os.path.isdir(offline_path):
            return offline_path
        return f"{model_base}/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"

    def _start_final_worker(self):
        if self._final_thread is not None:
            return
        self._final_running = True
        self._final_thread = threading.Thread(target=self._final_worker, daemon=True)
        self._final_thread.start()

    def _final_worker(self):
        while True:
            try:
                item = self._final_queue.get(timeout=0.1)
            except queue.Empty:
                if not self._final_running:
                    break
                continue
            if item is None:
                break
            segment_id, audio, start_time, end_time, fallback_text = item
            try:
                text = self._run_final_decode(audio)
                fallback_clean = fallback_text.strip()
                text_clean = text.strip()
                if fallback_clean:
                    min_len = max(4, int(len(fallback_clean) * 0.5))
                    if len(text_clean) < min_len:
                        text = fallback_text
                if self.config.use_punc and self._punc_model and text:
                    try:
                        with self._punc_lock:
                            punc_res = self._punc_model.generate(
                                input=text,
                                cache={},
                                disable_pbar=True,
                            )
                            if punc_res and len(punc_res) > 0 and 'text' in punc_res[0]:
                                text = punc_res[0]['text']
                    except Exception as e:
                        print(f"[ASREngine] Punctuation error: {e}")
                self.events.emit(
                    EventType.ASR_FINAL,
                    {
                        "text": text,
                        "confidence": 1.0,
                        "duration_ms": int((end_time - start_time) * 1000),
                    },
                    segment_id=segment_id
                )
            except Exception as e:
                print(f"[ASREngine] Final decode error: {e}")
                self._emit_fallback_final(segment_id, start_time, end_time, fallback_text)
            finally:
                self._final_queue.task_done()

    def _run_final_decode(self, audio: np.ndarray) -> str:
        model = self._final_asr_model or self._asr_model
        if self._final_is_streaming:
            res = model.generate(
                input=audio,
                cache={},
                is_final=True,
                chunk_size=self.config.asr_chunk_size,
                encoder_chunk_look_back=self.config.encoder_chunk_look_back,
                decoder_chunk_look_back=self.config.decoder_chunk_look_back,
                disable_pbar=True,
            )
        else:
            res = model.generate(
                input=audio,
                disable_pbar=True,
            )
        if res and len(res) > 0 and 'text' in res[0]:
            return res[0]['text']
        return ""

    def _emit_fallback_final(self, segment_id: str, start_time: float, end_time: float, full_text: str):
        if self.config.use_punc and self._punc_model and full_text:
            try:
                with self._punc_lock:
                    punc_res = self._punc_model.generate(
                        input=full_text,
                        cache={},
                        disable_pbar=True,
                    )
                    if punc_res and len(punc_res) > 0 and 'text' in punc_res[0]:
                        full_text = punc_res[0]['text']
            except Exception as e:
                print(f"[ASREngine] Punctuation error: {e}")
        self.events.emit(
            EventType.ASR_FINAL,
            {
                "text": full_text,
                "confidence": 1.0,
                "duration_ms": int((end_time - start_time) * 1000),
            },
            segment_id=segment_id
        )

    def finalize(self):
        """Finalize current session, process remaining audio"""
        if self._current_segment_id:
            self._end_segment()
        self._final_running = False
        try:
            self._final_queue.put_nowait(None)
        except queue.Full:
            pass

    # === Properties ===

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def is_speech_active(self) -> bool:
        return self._current_segment_id is not None

    @property
    def session_id(self) -> str:
        return self.events.session_id
