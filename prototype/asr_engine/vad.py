# -*- coding: utf-8 -*-
"""
VAD Processor for speech detection
Based on fsmn_vad_streaming model
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class VADConfig:
    """VAD configuration"""
    chunk_size_ms: int = 200  # milliseconds per chunk
    sample_rate: int = 16000
    min_speech_duration_ms: int = 200  # minimum speech duration to trigger
    min_silence_duration_ms: int = 300  # minimum silence to end segment

    @property
    def chunk_stride(self) -> int:
        """Samples per chunk"""
        return int(self.chunk_size_ms * self.sample_rate / 1000)


@dataclass
class SpeechSegment:
    """Detected speech segment"""
    start_ms: int
    end_ms: int  # -1 if still active
    is_active: bool = True

    @property
    def duration_ms(self) -> int:
        if self.end_ms < 0:
            return -1
        return self.end_ms - self.start_ms


class VADProcessor:
    """
    Voice Activity Detection processor using FunASR fsmn_vad model

    VAD output format:
    - [[beg, end]] - complete segment
    - [[beg, -1]] - speech started
    - [[-1, end]] - speech ended
    """

    def __init__(self, model, config: Optional[VADConfig] = None):
        self.model = model
        self.config = config or VADConfig()
        self.cache = {}
        self.current_segment: Optional[SpeechSegment] = None
        self._chunk_count = 0

    def reset(self):
        """Reset VAD state"""
        self.cache = {}
        self.current_segment = None
        self._chunk_count = 0

    def process_chunk(self, audio_chunk: np.ndarray, is_final: bool = False
                      ) -> Tuple[bool, Optional[SpeechSegment]]:
        """
        Process audio chunk through VAD

        Args:
            audio_chunk: Audio samples (float32)
            is_final: Whether this is the last chunk

        Returns:
            (is_speech_active, completed_segment or None)
        """
        # Run VAD inference
        res = self.model.generate(
            input=audio_chunk,
            cache=self.cache,
            is_final=is_final,
            chunk_size=self.config.chunk_size_ms,
            disable_pbar=True,
        )

        self._chunk_count += 1
        completed_segment = None

        # Parse VAD result
        if res and len(res) > 0 and 'value' in res[0]:
            segments = res[0]['value']

            for seg in segments:
                if len(seg) != 2:
                    continue

                beg, end = seg

                # Speech started: [[beg, -1]]
                if beg >= 0 and end < 0:
                    self.current_segment = SpeechSegment(
                        start_ms=beg,
                        end_ms=-1,
                        is_active=True
                    )

                # Speech ended: [[-1, end]]
                elif beg < 0 and end >= 0:
                    if self.current_segment:
                        self.current_segment.end_ms = end
                        self.current_segment.is_active = False
                        completed_segment = self.current_segment
                        self.current_segment = None

                # Complete segment: [[beg, end]]
                elif beg >= 0 and end >= 0:
                    completed_segment = SpeechSegment(
                        start_ms=beg,
                        end_ms=end,
                        is_active=False
                    )
                    self.current_segment = None

        is_speech_active = self.current_segment is not None and self.current_segment.is_active

        return is_speech_active, completed_segment

    @property
    def is_speech_active(self) -> bool:
        """Check if speech is currently active"""
        return self.current_segment is not None and self.current_segment.is_active
