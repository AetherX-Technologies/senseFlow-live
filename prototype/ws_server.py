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
import threading
import time
from datetime import datetime, timezone, timedelta
import numpy as np
import sounddevice as sd
from typing import Set, Any, Dict, List, Optional
import math

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

try:
    from pymongo import MongoClient
except ImportError:
    print("Installing pymongo...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymongo", "-q"])
    from pymongo import MongoClient

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


class MongoLogger:
    """Background MongoDB logger for session events."""

    def __init__(self, uri: str, db_name: str):
        self._client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        self._db = self._client[db_name]
        self._events = self._db["events"]
        self._queue: "queue.Queue[dict]" = queue.Queue()
        self._client.admin.command("ping")
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    @classmethod
    def from_env(cls) -> "MongoLogger | None":
        if os.getenv("MONGO_ENABLED", "1").lower() in {"0", "false", "no"}:
            return None
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGO_DB", "senseflow_live")
        try:
            return cls(uri, db_name)
        except Exception as e:
            print(f"{YELLOW}[Mongo]{RESET} Disabled (connection failed): {e}")
            return None

    def _worker(self) -> None:
        while self._running:
            item = self._queue.get()
            if item is None:
                break
            try:
                self._events.insert_one(item)
            except Exception as e:
                print(f"{YELLOW}[Mongo]{RESET} Write error: {e}")
            finally:
                self._queue.task_done()

    def log_event(self, session_id: str, event_type: str, payload: dict) -> None:
        doc = {
            "session_id": session_id,
            "event_type": event_type,
            "payload": payload,
            "created_at": datetime.now(timezone.utc)
        }
        self._queue.put(doc)

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        pipeline = [
            {
                "$group": {
                    "_id": "$session_id",
                    "first_at": {"$min": "$created_at"},
                    "last_at": {"$max": "$created_at"},
                    "event_count": {"$sum": 1},
                    "terminated_at": {
                        "$max": {
                            "$cond": [
                                {"$eq": ["$event_type", "session.terminated"]},
                                "$created_at",
                                None
                            ]
                        }
                    },
                }
            },
            {"$sort": {"last_at": -1}},
            {"$limit": int(limit)},
        ]
        sessions: List[Dict[str, Any]] = []
        for doc in self._events.aggregate(pipeline):
            first_at = doc.get("first_at")
            last_at = doc.get("last_at")
            terminated_at = doc.get("terminated_at")
            sessions.append({
                "session_id": doc.get("_id", ""),
                "started_at": first_at.timestamp() if first_at else None,
                "last_active": last_at.timestamp() if last_at else None,
                "event_count": int(doc.get("event_count", 0)),
                "terminated": bool(terminated_at),
                "terminated_at": terminated_at.timestamp() if terminated_at else None,
            })
        return sessions

    def list_sessions_missing_insights(self, limit: int = 5) -> List[str]:
        pipeline = [
            {
                "$group": {
                    "_id": "$session_id",
                    "has_final": {
                        "$max": {"$cond": [{"$eq": ["$event_type", "asr.final"]}, 1, 0]}
                    },
                    "has_insights": {
                        "$max": {"$cond": [{"$eq": ["$event_type", "insights.update"]}, 1, 0]}
                    },
                    "last_at": {"$max": "$created_at"},
                }
            },
            {"$match": {"has_final": 1, "has_insights": 0}},
            {"$sort": {"last_at": -1}},
            {"$limit": int(limit)},
        ]
        missing: List[str] = []
        for doc in self._events.aggregate(pipeline):
            session_id = doc.get("_id")
            if session_id:
                missing.append(session_id)
        return missing

    def get_session_data(self, session_id: str) -> Dict[str, Any]:
        transcript_docs = list(
            self._events.find(
                {"session_id": session_id, "event_type": "asr.final"}
            ).sort("created_at", 1)
        )
        transcript = []
        for doc in transcript_docs:
            payload = doc.get("payload", {})
            transcript.append({
                "segment_id": payload.get("segment_id", ""),
                "ts": payload.get("ts"),
                "text": payload.get("text", ""),
                "duration_ms": payload.get("duration_ms", 0),
            })

        insight_doc = self._events.find_one(
            {"session_id": session_id, "event_type": "insights.update"},
            sort=[("created_at", -1)],
        )
        insights = insight_doc.get("payload", {}) if insight_doc else {}

        qa_docs = list(
            self._events.find(
                {"session_id": session_id, "event_type": "qa.answer"}
            ).sort("created_at", 1)
        )
        qa = []
        for doc in qa_docs:
            payload = doc.get("payload", {})
            created_at = doc.get("created_at")
            ts_ms = int(created_at.timestamp() * 1000) if created_at else None
            qa.append({
                "question": payload.get("question", ""),
                "answer": payload.get("answer", ""),
                "ts_ms": ts_ms,
            })

        terminated_doc = self._events.find_one(
            {"session_id": session_id, "event_type": "session.terminated"},
            sort=[("created_at", -1)],
        )
        terminated_at = terminated_doc.get("created_at") if terminated_doc else None

        return {
            "session_id": session_id,
            "transcript": transcript,
            "insights": insights,
            "qa": qa,
            "terminated": bool(terminated_at),
            "terminated_at": terminated_at.timestamp() if terminated_at else None,
        }

    def close(self) -> None:
        self._running = False
        self._queue.put(None)
        self._thread.join(timeout=2)
        self._client.close()

    def apply_retention(self, retention_days: int, max_size_mb: int) -> Dict[str, Any]:
        deleted_by_age = 0
        deleted_by_size = 0
        size_before = 0
        size_after = 0

        try:
            stats_before = self._db.command("collStats", "events")
            size_before = int(stats_before.get("size", 0))
            avg_obj_size = max(1, int(stats_before.get("avgObjSize", 1)))
        except Exception:
            avg_obj_size = 1

        if retention_days and retention_days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=int(retention_days))
            result = self._events.delete_many({"created_at": {"$lt": cutoff}})
            deleted_by_age = result.deleted_count

        try:
            stats_after_age = self._db.command("collStats", "events")
            size_after = int(stats_after_age.get("size", 0))
        except Exception:
            size_after = size_before

        if max_size_mb and max_size_mb > 0:
            limit_bytes = int(max_size_mb) * 1024 * 1024
            if size_after > limit_bytes:
                excess = size_after - limit_bytes
                docs_to_delete = int(math.ceil(excess / avg_obj_size))
                if docs_to_delete > 0:
                    ids = [
                        doc["_id"]
                        for doc in self._events.find({}, {"_id": 1}).sort("created_at", 1).limit(docs_to_delete)
                    ]
                    if ids:
                        result = self._events.delete_many({"_id": {"$in": ids}})
                        deleted_by_size = result.deleted_count
                try:
                    stats_final = self._db.command("collStats", "events")
                    size_after = int(stats_final.get("size", 0))
                except Exception:
                    pass

        return {
            "deleted_by_age": deleted_by_age,
            "deleted_by_size": deleted_by_size,
            "size_before": size_before,
            "size_after": size_after,
        }


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

        # MongoDB logging
        self.mongo = MongoLogger.from_env()

        # Settings
        self.settings = {
            "audio": {
                "micId": "",
                "gain": 1.0,
                "noiseGate": 0.0,
                "vadSensitivity": 0.5,
            },
            "transcription": {
                "punctuation": True,
                "mergeStrategy": "silence",
                "modelMode": "realtime",
            },
            "summary": {
                "intervalSec": 180,
                "liveSummary": True,
                "llmEnabled": True,
                "llmModel": self.llm.config.model,
            },
            "display": {
                "autoScroll": True,
                "showTimestamps": True,
                "exportFormat": "markdown",
            },
            "storage": {
                "mongoEnabled": True,
                "retentionDays": 30,
                "maxSizeMb": 2048,
                "autoCleanup": True,
            },
        }
        self.audio_gain = 1.0
        self.noise_gate = 0.0
        self.vad_sensitivity = 0.5
        self.llm_enabled = True
        self.insights.set_summary_interval(self.settings["summary"]["intervalSec"])

        # Audio stream state
        self._audio_stream = None
        self.audio_device = None
        self._cleanup_task = None
        self.audio_paused = False
        self._repair_interval_sec = int(os.getenv("INSIGHTS_REPAIR_INTERVAL_SEC", "300"))
        self._repair_limit = int(os.getenv("INSIGHTS_REPAIR_LIMIT", "2"))
        self._repair_task = None
        self._repair_lock: Optional[asyncio.Lock] = None
        self._live_seeded = False
        self._terminated_sessions: Dict[str, float] = {}
        self._paused_by_terminate = False

    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        if not self.clients:
            return
        msg_str = json.dumps(message, ensure_ascii=False)
        await asyncio.gather(
            *[client.send(msg_str) for client in self.clients],
            return_exceptions=True
        )

    @staticmethod
    def _has_insights(insights: Optional[Dict[str, Any]]) -> bool:
        if not insights:
            return False
        return bool(
            insights.get("summary")
            or insights.get("summary_live")
            or insights.get("actions")
            or insights.get("questions")
        )

    def _is_session_terminated(self, session_id: Optional[str]) -> bool:
        if not session_id:
            return False
        return session_id in self._terminated_sessions

    def _get_terminated_at(self, session_id: Optional[str]) -> Optional[float]:
        if not session_id:
            return None
        return self._terminated_sessions.get(session_id)

    def _mark_session_terminated(self, session_id: str, terminated_at: Optional[float] = None) -> float:
        ts = terminated_at or time.time()
        self._terminated_sessions[session_id] = ts
        return ts

    @staticmethod
    def _compose_transcript(items: List[Dict[str, Any]]) -> str:
        parts = [
            item.get("text", "").strip()
            for item in items
            if item.get("text")
        ]
        return " ".join(parts).strip()

    def _on_event(self, event: ASREvent):
        """Handle ASR event and queue for broadcast"""
        if self._loop:
            self._loop.call_soon_threadsafe(
                lambda e=event: asyncio.create_task(self._handle_event(e))
            )

    async def _handle_event(self, event: ASREvent):
        """Process ASR event and broadcast"""
        if self._is_session_terminated(event.session_id):
            return
        # Broadcast the event
        await self.broadcast(event.to_dict())

        # On final text, add to transcript and maybe generate insights
        if event.type == EventType.ASR_FINAL:
            text = event.payload.get("text", "")
            if text:
                self.insights.add_text(text)
                print(f"{CYAN}[LLM]{RESET} Added {len(text)} chars, total: {len(self.insights.full_transcript)}")

                if self.mongo:
                    self.mongo.log_event(
                        event.session_id,
                        "asr.final",
                        {
                            "segment_id": event.segment_id,
                            "ts": event.ts,
                            "text": text,
                            "duration_ms": event.payload.get("duration_ms", 0)
                        }
                    )

                # Check if we should generate insights
                if self.llm_enabled and self.insights.should_generate_summary():
                    asyncio.create_task(self._generate_and_send_insights())

    async def _generate_and_send_insights(self, force: bool = False):
        """Generate insights and send to clients"""
        try:
            if not self.llm_enabled:
                return
            session_id = self.engine.session_id
            if self._is_session_terminated(session_id):
                return
            print(f"{CYAN}[LLM]{RESET} Generating insights...")
            insights = await self.insights.generate_insights(force=force)

            summary_live = insights.get("summary_live", []) if insights else []
            if insights and (insights.get("summary") or summary_live or insights.get("actions") or insights.get("questions")):
                if session_id != self.engine.session_id:
                    print(f"{GRAY}[LLM]{RESET} Skipped stale insights for session {session_id}")
                    return
                if self.mongo:
                    self.mongo.log_event(
                        session_id,
                        "insights.update",
                        insights
                    )
                await self.broadcast({
                    "type": "insights.update",
                    "session_id": session_id,
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

    async def _send_history_list(
        self,
        websocket: Optional[websockets.WebSocketServerProtocol] = None,
        limit: int = 50,
    ):
        try:
            safe_limit = int(limit)
        except (TypeError, ValueError):
            safe_limit = 50
        sessions: List[Dict[str, Any]] = []
        mongo_enabled = self.mongo is not None
        if self.mongo:
            try:
                sessions = await asyncio.to_thread(self.mongo.list_sessions, safe_limit)
            except Exception as e:
                print(f"{YELLOW}[Mongo]{RESET} List error: {e}")
                sessions = []

        live_session_id = self.engine.session_id if self.engine else ""
        if live_session_id and not any(s.get("session_id") == live_session_id for s in sessions):
            sessions.insert(0, {
                "session_id": live_session_id,
                "started_at": None,
                "last_active": None,
                "event_count": 0,
                "source": "live",
                "terminated": self._is_session_terminated(live_session_id),
                "terminated_at": self._get_terminated_at(live_session_id),
            })

        if self._terminated_sessions:
            session_index = {item.get("session_id"): item for item in sessions}
            for session_id, terminated_at in self._terminated_sessions.items():
                meta = session_index.get(session_id)
                if not meta:
                    meta = {
                        "session_id": session_id,
                        "started_at": None,
                        "last_active": None,
                        "event_count": 0,
                    }
                    sessions.append(meta)
                    session_index[session_id] = meta
                meta["terminated"] = True
                meta["terminated_at"] = terminated_at

        message = {
            "type": "history.list",
            "payload": {
                "sessions": sessions,
                "live_session_id": live_session_id,
                "mongo_enabled": mongo_enabled,
            }
        }
        if websocket:
            await websocket.send(json.dumps(message, ensure_ascii=False))
        else:
            await self.broadcast(message)

    async def _send_history_session(self, websocket: websockets.WebSocketServerProtocol, session_id: str):
        data = {
            "session_id": session_id,
            "transcript": [],
            "insights": {},
            "qa": [],
        }
        if self.mongo:
            try:
                data = await asyncio.to_thread(self.mongo.get_session_data, session_id)
            except Exception as e:
                print(f"{YELLOW}[Mongo]{RESET} Load error: {e}")
        if self.engine and session_id == self.engine.session_id:
            current = self.insights.current_insights() if self.insights else {}
            if self._has_insights(current) and not self._has_insights(data.get("insights")):
                data["insights"] = current
                if self.mongo:
                    self.mongo.log_event(session_id, "insights.update", current)
        if self._is_session_terminated(session_id):
            data["terminated"] = True
            if not data.get("terminated_at"):
                data["terminated_at"] = self._get_terminated_at(session_id)

        message = {
            "type": "history.session",
            "payload": data
        }
        await websocket.send(json.dumps(message, ensure_ascii=False))

    async def _run_cleanup(self):
        if not self.mongo:
            return
        storage = self.settings.get("storage", {})
        if not storage.get("autoCleanup", False):
            return
        retention_days = int(storage.get("retentionDays", 0) or 0)
        max_size_mb = int(storage.get("maxSizeMb", 0) or 0)
        try:
            result = await asyncio.to_thread(self.mongo.apply_retention, retention_days, max_size_mb)
            print(f"{CYAN}[Mongo]{RESET} Cleanup: {result}")
        except Exception as e:
            print(f"{YELLOW}[Mongo]{RESET} Cleanup error: {e}")

    async def _cleanup_loop(self):
        while self.running:
            await asyncio.sleep(300)
            await self._run_cleanup()

    async def _repair_missing_insights(self):
        if not self.mongo or not self.llm_enabled:
            return
        if self._repair_lock is None:
            self._repair_lock = asyncio.Lock()
        if self._repair_lock.locked():
            return
        async with self._repair_lock:
            try:
                missing = await asyncio.to_thread(
                    self.mongo.list_sessions_missing_insights,
                    self._repair_limit
                )
            except Exception as e:
                print(f"{YELLOW}[Mongo]{RESET} Repair list error: {e}")
                return
            if not missing:
                return
            for session_id in missing:
                if self.engine and session_id == self.engine.session_id:
                    continue
                try:
                    data = await asyncio.to_thread(self.mongo.get_session_data, session_id)
                except Exception as e:
                    print(f"{YELLOW}[Mongo]{RESET} Repair load error: {e}")
                    continue
                transcript_items = data.get("transcript", [])
                transcript = " ".join(
                    item.get("text", "") for item in transcript_items if item.get("text")
                ).strip()
                if not transcript:
                    continue
                try:
                    insights = await self.llm.generate_summary(transcript, previous=None, force=True)
                except Exception as e:
                    print(f"{YELLOW}[LLM]{RESET} Repair failed for {session_id}: {e}")
                    continue
                if not self._has_insights(insights):
                    continue
                self.mongo.log_event(session_id, "insights.update", insights)

    async def _repair_loop(self):
        while self.running:
            await asyncio.sleep(self._repair_interval_sec)
            await self._repair_missing_insights()

    def _merge_settings(self, base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_settings(merged.get(key, {}), value)
            else:
                merged[key] = value
        return merged

    def _apply_settings(self, incoming: Dict[str, Any]) -> Dict[str, Any]:
        applied: List[str] = []
        ignored: List[str] = []
        requires_restart: List[str] = []
        cleanup_requested = False

        if not incoming or not isinstance(incoming, dict):
            return {
                "applied": applied,
                "ignored": ["settings"],
                "requires_restart": requires_restart,
                "cleanup_requested": cleanup_requested,
            }

        self.settings = self._merge_settings(self.settings, incoming)

        audio = self.settings.get("audio", {})
        gain = audio.get("gain", self.audio_gain)
        noise_gate = audio.get("noiseGate", self.noise_gate)
        vad_sensitivity = audio.get("vadSensitivity", self.vad_sensitivity)
        mic_id = audio.get("micId")

        try:
            self.audio_gain = max(0.1, min(float(gain), 4.0))
            applied.append("audio.gain")
        except (TypeError, ValueError):
            ignored.append("audio.gain")

        try:
            gate_value = max(0.0, min(float(noise_gate), 1.0))
            self.noise_gate = gate_value * 0.05
            applied.append("audio.noiseGate")
        except (TypeError, ValueError):
            ignored.append("audio.noiseGate")

        try:
            self.vad_sensitivity = max(0.0, min(float(vad_sensitivity), 1.0))
            applied.append("audio.vadSensitivity")
            if self.engine:
                min_padding = 300
                max_padding = 1200
                padding = int(max_padding - self.vad_sensitivity * (max_padding - min_padding))
                self.engine.config.vad_end_padding_ms = padding
        except (TypeError, ValueError):
            ignored.append("audio.vadSensitivity")

        resolved_device = self._resolve_input_device(mic_id)
        if resolved_device != self.audio_device:
            try:
                if self.audio_paused:
                    self.audio_device = resolved_device
                else:
                    self._start_audio_stream(resolved_device)
                applied.append("audio.micId")
            except Exception as e:
                ignored.append("audio.micId")
                print(f"{YELLOW}[Audio]{RESET} Failed to switch mic: {e}")

        transcription = self.settings.get("transcription", {})
        if "punctuation" in transcription:
            if self.engine:
                self.engine.config.use_punc = bool(transcription.get("punctuation"))
                applied.append("transcription.punctuation")
            else:
                ignored.append("transcription.punctuation")
        if "mergeStrategy" in transcription:
            ignored.append("transcription.mergeStrategy")
        if "modelMode" in transcription:
            mode = transcription.get("modelMode")
            if self.engine and mode in {"realtime", "offline"}:
                self.engine.set_streaming_enabled(mode == "realtime")
                applied.append("transcription.modelMode")
            else:
                ignored.append("transcription.modelMode")

        summary = self.settings.get("summary", {})
        if "intervalSec" in summary:
            self.insights.set_summary_interval(summary.get("intervalSec"))
            applied.append("summary.intervalSec")
        if "llmEnabled" in summary:
            self.llm_enabled = bool(summary.get("llmEnabled"))
            applied.append("summary.llmEnabled")
        if "llmModel" in summary:
            model = summary.get("llmModel")
            if model:
                self.llm.config.model = model
                applied.append("summary.llmModel")
        if "liveSummary" in summary:
            ignored.append("summary.liveSummary")

        display = self.settings.get("display", {})
        if "autoScroll" in display:
            ignored.append("display.autoScroll")
        if "showTimestamps" in display:
            ignored.append("display.showTimestamps")
        if "exportFormat" in display:
            ignored.append("display.exportFormat")

        storage = self.settings.get("storage", {})
        if "mongoEnabled" in storage:
            want_mongo = bool(storage.get("mongoEnabled"))
            if want_mongo and not self.mongo:
                os.environ["MONGO_ENABLED"] = "1"
                self.mongo = MongoLogger.from_env()
            if not want_mongo and self.mongo:
                os.environ["MONGO_ENABLED"] = "0"
                self.mongo.close()
                self.mongo = None
            applied.append("storage.mongoEnabled")
        if "retentionDays" in storage:
            cleanup_requested = True
            applied.append("storage.retentionDays")
        if "maxSizeMb" in storage:
            cleanup_requested = True
            applied.append("storage.maxSizeMb")
        if "autoCleanup" in storage:
            applied.append("storage.autoCleanup")

        return {
            "applied": applied,
            "ignored": ignored,
            "requires_restart": requires_restart,
            "cleanup_requested": cleanup_requested,
        }

    def _audio_callback(self, indata, frames, time_info, status):
        """Audio capture callback"""
        if status:
            print(f"{GRAY}Audio: {status}{RESET}", file=sys.stderr)
        audio = indata[:, 0].copy()
        if self.audio_gain != 1.0:
            audio = audio * self.audio_gain
            audio = np.clip(audio, -1.0, 1.0)
        if self.noise_gate > 0:
            audio[np.abs(audio) < self.noise_gate] = 0.0
        self.audio_queue.put(audio)

    def _create_audio_stream(self, device_index: Optional[int]):
        return sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=BLOCK_SIZE,
            device=device_index,
            callback=self._audio_callback
        )

    def _start_audio_stream(self, device_index: Optional[int]):
        if self._audio_stream:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception:
                pass
        self._audio_stream = self._create_audio_stream(device_index)
        self._audio_stream.start()
        self.audio_device = device_index

    def _stop_audio_stream(self):
        if not self._audio_stream:
            return
        try:
            self._audio_stream.stop()
            self._audio_stream.close()
        except Exception:
            pass
        self._audio_stream = None

    def _clear_audio_queue(self):
        try:
            while True:
                self.audio_queue.get_nowait()
        except queue.Empty:
            return

    def _set_audio_paused(self, paused: bool):
        self.audio_paused = paused
        if paused:
            self._stop_audio_stream()
            self._clear_audio_queue()
        else:
            self._start_audio_stream(self.audio_device)

    def _resolve_input_device(self, mic_id: Optional[str]) -> Optional[int]:
        if mic_id is None:
            return None
        mic_id = str(mic_id).strip()
        if not mic_id:
            return None
        try:
            device_index = int(mic_id)
            info = sd.query_devices(device_index)
            if info and info.get("max_input_channels", 0) > 0:
                return device_index
        except (ValueError, TypeError):
            device_index = None
        try:
            devices = sd.query_devices()
        except Exception:
            return None
        lower_target = mic_id.lower()
        for index, device in enumerate(devices):
            if device.get("max_input_channels", 0) <= 0:
                continue
            name = str(device.get("name", "")).lower()
            if lower_target in name or name in lower_target:
                return index
        return None

    def _list_audio_devices(self) -> List[Dict[str, Any]]:
        devices = []
        try:
            all_devices = sd.query_devices()
            default_input = sd.default.device[0]
        except Exception:
            return devices
        for index, device in enumerate(all_devices):
            if device.get("max_input_channels", 0) <= 0:
                continue
            devices.append({
                "id": str(index),
                "name": device.get("name", f"Audio {index}"),
                "is_default": index == default_input,
            })
        return devices

    async def _process_audio(self):
        """Process audio in async loop"""
        while self.running:
            try:
                if self.audio_paused:
                    self._clear_audio_queue()
                    await asyncio.sleep(0.05)
                    continue
                # Non-blocking check for audio
                try:
                    audio_chunk = self.audio_queue.get_nowait()
                    self.engine.feed_audio(audio_chunk)
                except queue.Empty:
                    pass
                await asyncio.sleep(0.01)  # Small delay to prevent CPU spin
            except Exception as e:
                print(f"{YELLOW}[Audio] Error: {e}{RESET}")

    async def _seed_live_from_mongo(self) -> None:
        if self._live_seeded or not self.mongo or not self.engine or not self.insights:
            return
        session_id = self.engine.session_id
        try:
            data = await asyncio.to_thread(self.mongo.get_session_data, session_id)
        except Exception as e:
            print(f"{YELLOW}[Mongo]{RESET} Seed error: {e}")
            return
        transcript_items = data.get("transcript", [])
        transcript_parts = [
            item.get("text", "")
            for item in transcript_items
            if item.get("text")
        ]
        if not transcript_parts:
            return
        self.insights.seed_from_history(transcript_parts, data.get("insights"))
        self._live_seeded = True

    async def _answer_question_for_session(self, session_id: str, question: str) -> str:
        transcript = ""
        insights: Dict[str, Any] = {}
        is_live = self.engine and session_id == self.engine.session_id

        if is_live and not self._is_session_terminated(session_id) and self.insights:
            await self._seed_live_from_mongo()
            transcript = self.insights.full_transcript
            insights = self.insights.current_insights()
            if transcript or self._has_insights(insights):
                return await self.llm.answer_question(question, transcript, insights)

        if self.mongo:
            try:
                data = await asyncio.to_thread(self.mongo.get_session_data, session_id)
            except Exception as e:
                print(f"{YELLOW}[Mongo]{RESET} QA load error: {e}")
                data = {}
            transcript = self._compose_transcript(data.get("transcript", []))
            insights = data.get("insights", {}) or {}

        if not transcript and is_live and self.insights:
            transcript = self.insights.full_transcript
            if not self._has_insights(insights):
                insights = self.insights.current_insights()

        return await self.llm.answer_question(question, transcript, insights)

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
        await websocket.send(json.dumps({
            "type": "audio.paused",
            "payload": {"paused": self.audio_paused}
        }, ensure_ascii=False))
        if self._is_session_terminated(self.engine.session_id):
            await websocket.send(json.dumps({
                "type": "session.terminated",
                "payload": {
                    "session_id": self.engine.session_id,
                    "terminated_at": self._get_terminated_at(self.engine.session_id),
                }
            }, ensure_ascii=False))
        current_insights = self.insights.current_insights() if self.insights else {}
        if self._has_insights(current_insights):
            await websocket.send(json.dumps({
                "type": "insights.update",
                "session_id": self.engine.session_id,
                "payload": current_insights
            }, ensure_ascii=False))

        try:
            async for message in websocket:
                # Handle client messages
                try:
                    data = json.loads(message)
                    cmd = data.get("command")

                    if cmd == "reset":
                        old_session_id = self.engine.session_id
                        self.engine.reset()
                        new_session_id = self.engine.new_session()
                        self.insights = InsightGenerator(self.llm)
                        self.insights.set_summary_interval(self.settings["summary"]["intervalSec"])
                        self._live_seeded = False
                        if new_session_id in self._terminated_sessions:
                            self._terminated_sessions.pop(new_session_id, None)
                        if self._paused_by_terminate:
                            self._set_audio_paused(False)
                            self._paused_by_terminate = False
                            await self.broadcast({
                                "type": "audio.paused",
                                "payload": {"paused": self.audio_paused}
                            })
                        if self.mongo:
                            self.mongo.log_event(
                                old_session_id,
                                "session.reset",
                                {"next_session_id": new_session_id}
                            )
                            self.mongo.log_event(
                                new_session_id,
                                "session.start",
                                {"host": self.host, "port": self.port, "reason": "reset"}
                            )
                        await self.broadcast({
                            "type": "session.changed",
                            "payload": {
                                "session_id": new_session_id,
                                "previous_session_id": old_session_id,
                                "reason": "reset"
                            }
                        })
                        await websocket.send(json.dumps({
                            "type": "engine.reset",
                            "payload": {"status": "ok", "session_id": new_session_id}
                        }, ensure_ascii=False))

                    elif cmd == "session.terminate":
                        session_id = data.get("session_id") or self.engine.session_id
                        if session_id:
                            terminated_at = self._get_terminated_at(session_id)
                            if not terminated_at:
                                terminated_at = self._mark_session_terminated(session_id)
                                if self.mongo:
                                    self.mongo.log_event(
                                        session_id,
                                        "session.terminated",
                                        {"reason": "manual", "terminated_at": terminated_at}
                                    )
                            await self.broadcast({
                                "type": "session.terminated",
                                "payload": {
                                    "session_id": session_id,
                                    "terminated_at": terminated_at,
                                }
                            })
                            if session_id == self.engine.session_id:
                                if not self.audio_paused:
                                    self._set_audio_paused(True)
                                    self._paused_by_terminate = True
                                else:
                                    self._paused_by_terminate = False
                                await self.broadcast({
                                    "type": "audio.paused",
                                    "payload": {"paused": self.audio_paused}
                                })

                    elif cmd == "ask":
                        # Handle Q&A
                        question = (data.get("question") or "").strip()
                        target_session_id = data.get("session_id") or self.engine.session_id
                        if question:
                            print(f"{CYAN}[QA]{RESET} Question: {question}")
                            if not self.llm_enabled:
                                await websocket.send(json.dumps({
                                    "type": "qa.answer",
                                    "session_id": target_session_id,
                                    "payload": {
                                        "question": question,
                                        "answer": "LLM 已关闭，无法回答。"
                                    }
                                }, ensure_ascii=False))
                                continue
                            answer = await self._answer_question_for_session(target_session_id, question)
                            if self.mongo:
                                self.mongo.log_event(
                                    target_session_id,
                                    "qa.answer",
                                    {"question": question, "answer": answer}
                                )
                            await websocket.send(json.dumps({
                                "type": "qa.answer",
                                "session_id": target_session_id,
                                "payload": {
                                    "question": question,
                                    "answer": answer
                                }
                            }, ensure_ascii=False))
                            print(f"{GREEN}[QA]{RESET} Answered: {answer[:50]}...")

                    elif cmd == "generate_insights":
                        # Force generate insights
                        asyncio.create_task(self._generate_and_send_insights())

                    elif cmd == "history.list":
                        limit = data.get("limit", 50)
                        await self._send_history_list(websocket, limit=limit)

                    elif cmd == "history.load":
                        session_id = data.get("session_id")
                        if session_id:
                            await self._send_history_session(websocket, session_id)

                    elif cmd == "settings.update":
                        incoming = data.get("settings", {})
                        result = self._apply_settings(incoming)
                        if result.get("cleanup_requested"):
                            asyncio.create_task(self._run_cleanup())
                        await websocket.send(json.dumps({
                            "type": "settings.applied",
                            "payload": result
                        }, ensure_ascii=False))

                    elif cmd == "audio.devices":
                        devices = self._list_audio_devices()
                        await websocket.send(json.dumps({
                            "type": "audio.devices",
                            "payload": {
                                "devices": devices,
                                "selected": str(self.audio_device) if self.audio_device is not None else ""
                            }
                        }, ensure_ascii=False))

                    elif cmd == "audio.pause":
                        requested = data.get("paused")
                        was_paused = self.audio_paused
                        if requested is None:
                            requested = not self.audio_paused
                        if self._is_session_terminated(self.engine.session_id):
                            if not self.audio_paused:
                                self._set_audio_paused(True)
                            await self.broadcast({
                                "type": "audio.paused",
                                "payload": {"paused": self.audio_paused}
                            })
                            continue
                        try:
                            self._set_audio_paused(bool(requested))
                        except Exception as e:
                            print(f"{YELLOW}[Audio]{RESET} Pause error: {e}")
                        await self.broadcast({
                            "type": "audio.paused",
                            "payload": {"paused": self.audio_paused}
                        })
                        if (not was_paused) and self.audio_paused and self.llm_enabled and not self._is_session_terminated(self.engine.session_id):
                            asyncio.create_task(self._generate_and_send_insights(force=True))

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

        if self.mongo:
            self.mongo.log_event(
                self.engine.session_id,
                "session.start",
                {"host": self.host, "port": self.port}
            )

        # Test LLM connection
        print(f"\n[2/3] Testing LLM connection...")
        print(f"  Model: {self.llm.config.model}")
        print(f"  URL: {self.llm.config.base_url}")

        # Start audio stream
        print(f"\n[3/3] Starting audio capture...")
        device_index = self.audio_device if self.audio_device is not None else sd.default.device[0]
        device_info = sd.query_devices(device_index)
        print(f"  Microphone: {device_info['name']}")

        self.audio_paused = False
        self._start_audio_stream(self.audio_device)

        self.running = True
        self._loop = asyncio.get_running_loop()
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        if self.mongo and self._repair_interval_sec > 0:
            self._repair_task = asyncio.create_task(self._repair_loop())

        print(f"\n{'-' * 60}")
        print(f"WebSocket: ws://{self.host}:{self.port}")
        print(f"Open prototype/index_live.html in browser")
        print(f"Press {GREEN}Ctrl+C{RESET} to stop")
        print(f"{'-' * 60}\n")

        # Start WebSocket server
        async with serve(self.handler, self.host, self.port):
            # Run audio processing
            await self._process_audio()

    async def cleanup(self):
        """Cleanup resources"""
        await self.llm.close()
        if self.mongo:
            self.mongo.close()

    def stop(self):
        """Stop the server"""
        self.running = False
        if self.engine:
            self.engine.finalize()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._repair_task:
            self._repair_task.cancel()
        self._stop_audio_stream()


async def main():
    server = ASRWebSocketServer(host="127.0.0.1", port=8766)
    try:
        await server.start()
    except KeyboardInterrupt:
        print(f"\n{GREEN}Shutting down...{RESET}")
        server.stop()
        await server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
