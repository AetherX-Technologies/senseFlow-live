# -*- coding: utf-8 -*-
"""
Event system for ASR Engine
Based on docs/references/api-spec.md
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import time
import uuid


class EventType(Enum):
    """Event types as defined in api-spec.md"""
    # Audio events
    AUDIO_SEGMENT_STARTED = "audio.segment.started"
    AUDIO_SEGMENT_ENDED = "audio.segment.ended"

    # ASR events
    ASR_PARTIAL = "asr.partial"
    ASR_FINAL = "asr.final"

    # VAD events
    VAD_SPEECH_START = "vad.speech.start"
    VAD_SPEECH_END = "vad.speech.end"

    # Engine events
    ENGINE_READY = "engine.ready"
    ENGINE_ERROR = "engine.error"


@dataclass
class ASREvent:
    """Event envelope as defined in api-spec.md"""
    type: EventType
    payload: Dict[str, Any]
    ts: float = field(default_factory=time.time)
    session_id: str = ""
    segment_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "ts": self.ts,
            "session_id": self.session_id,
            "segment_id": self.segment_id,
            "payload": self.payload
        }


class EventEmitter:
    """Simple event emitter for ASR events"""

    def __init__(self):
        self._listeners: Dict[EventType, List[Callable[[ASREvent], None]]] = {}
        self._session_id = str(uuid.uuid4())[:8]

    @property
    def session_id(self) -> str:
        return self._session_id

    def on(self, event_type: EventType, callback: Callable[[ASREvent], None]):
        """Register event listener"""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)

    def off(self, event_type: EventType, callback: Callable[[ASREvent], None]):
        """Remove event listener"""
        if event_type in self._listeners:
            self._listeners[event_type].remove(callback)

    def emit(self, event_type: EventType, payload: Dict[str, Any],
             segment_id: str = "") -> ASREvent:
        """Emit event to all listeners"""
        event = ASREvent(
            type=event_type,
            payload=payload,
            session_id=self._session_id,
            segment_id=segment_id
        )

        if event_type in self._listeners:
            for callback in self._listeners[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"[EventEmitter] Error in listener: {e}")

        return event
