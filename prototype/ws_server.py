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
import inspect
from typing import Set, Any, Dict, List, Optional
import math
from dataclasses import replace
try:
    import soundcard as sc
except ImportError:
    sc = None

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
        self.system_engine: Optional[ASREngine] = None
        self._mic_queue = queue.Queue()
        self._system_queue = queue.Queue()
        self.running = False
        self._loop = None
        self._engine_ready = False
        self._init_task: Optional[asyncio.Task] = None
        self._init_error: Optional[str] = None

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
        try:
            sys_max_segment_ms = int(os.getenv("SYS_MAX_SEGMENT_MS", "15000"))
        except ValueError:
            sys_max_segment_ms = 15000
        if sys_max_segment_ms < 0:
            sys_max_segment_ms = 0
        self.settings = {
            "audio": {
                "source": "mic",
                "micId": "",
                "systemDevice": "",
                "gain": 1.0,
                "noiseGate": 0.0,
                "vadSensitivity": 0.5,
                "sysMaxSegmentMs": sys_max_segment_ms,
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
        self._system_stream = None
        self._system_thread = None
        self._system_stop = threading.Event()
        self._system_sample_rate = SAMPLE_RATE
        self.audio_device = None
        self.system_device = None
        self.audio_source = "mic"
        self._wasapi_loopback = self._supports_wasapi_loopback()
        self._cleanup_task = None
        self.audio_paused = False
        self._repair_interval_sec = int(os.getenv("INSIGHTS_REPAIR_INTERVAL_SEC", "300"))
        self._repair_limit = int(os.getenv("INSIGHTS_REPAIR_LIMIT", "2"))
        self._repair_task = None
        self._repair_lock: Optional[asyncio.Lock] = None
        self._live_seeded = False
        self._terminated_sessions: Dict[str, float] = {}
        self._paused_by_terminate = False
        self._recording_stats: Dict[str, Dict[str, Any]] = {}
        self._aec_delay_samples = 0
        self._aec_last_update = 0.0
        self._aec_min_sys_rms = 0.004
        self._aec_min_corr = 0.15
        self._aec_max_delay_ms = 80
        self._aec_spec_floor = 1e-3
        self._nlms_enabled = os.getenv("NLMS_ENABLE", "0") == "1"
        self._nlms_order = int(os.getenv("NLMS_ORDER", "512"))
        self._nlms_mu = float(os.getenv("NLMS_MU", "0.3"))
        self._nlms_eps = float(os.getenv("NLMS_EPS", "1e-3"))
        self._nlms_w = np.zeros(self._nlms_order, dtype=np.float32)
        self._silence_rms = 1e-5

    async def _initialize_engines(self) -> None:
        """Initialize ASR engines without blocking the event loop."""
        try:
            print(f"{CYAN}[Init]{RESET} Starting ASR model initialization...")
            await self.broadcast({
                "type": "engine.status",
                "payload": {"ready": False, "loading": True}
            })
            if not self.engine or not self.system_engine:
                return
            if not await asyncio.to_thread(self.engine.initialize):
                self._init_error = "Failed to initialize ASR Engine"
                await self.broadcast({
                    "type": "engine.status",
                    "payload": {"ready": False, "error": self._init_error}
                })
                return
            if not await asyncio.to_thread(self.system_engine.initialize):
                self._init_error = "Failed to initialize system ASR Engine"
                await self.broadcast({
                    "type": "engine.status",
                    "payload": {"ready": False, "error": self._init_error}
                })
                return
            self.system_engine.new_session()

            if self.mongo:
                self.mongo.log_event(
                    self.engine.session_id,
                    "session.start",
                    {"host": self.host, "port": self.port}
                )

            print(f"\n[2/3] Testing LLM connection...")
            print(f"  Model: {self.llm.config.model}")
            print(f"  URL: {self.llm.config.base_url}")

            print(f"\n[3/3] Starting audio capture...")
            if self.audio_device is None:
                self.audio_device = sd.default.device[0]
            if self.system_device is None:
                self.system_device = sd.default.device[1]
            if self.audio_source in {"mic", "both"}:
                device_info = sd.query_devices(self.audio_device)
                print(f"  Microphone: {device_info['name']}")
            if self.audio_source in {"system", "both"}:
                out_info = sd.query_devices(self.system_device)
                print(f"  System audio: {out_info['name']}")

            self.audio_paused = False
            self._refresh_audio_streams()
            self._set_recording_state(self.engine.session_id, True)
            self._engine_ready = True
            await self.broadcast({
                "type": "engine.status",
                "payload": {"ready": True}
            })
        except Exception as e:
            self._init_error = str(e)
            print(f"{YELLOW}[Init]{RESET} Failed: {e}")
            await self.broadcast({
                "type": "engine.status",
                "payload": {"ready": False, "error": self._init_error}
            })

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

    def _ensure_recording_stats(self, session_id: str) -> Dict[str, Any]:
        stats = self._recording_stats.get(session_id)
        if not stats:
            stats = {
                "recorded_ms": 0,
                "recording": False,
                "last_resume_ts": None,
            }
            self._recording_stats[session_id] = stats
        return stats

    def _set_recording_state(self, session_id: str, recording: bool) -> Dict[str, Any]:
        stats = self._ensure_recording_stats(session_id)
        now = time.time()
        if recording:
            if not stats["recording"]:
                stats["recording"] = True
                stats["last_resume_ts"] = now
        else:
            if stats["recording"]:
                last = stats.get("last_resume_ts") or now
                stats["recorded_ms"] += int((now - last) * 1000)
                stats["recording"] = False
                stats["last_resume_ts"] = None
        return stats

    def _get_recorded_ms(self, session_id: str) -> int:
        stats = self._ensure_recording_stats(session_id)
        total = int(stats.get("recorded_ms", 0))
        if stats.get("recording") and stats.get("last_resume_ts"):
            total += int((time.time() - stats["last_resume_ts"]) * 1000)
        return total

    def _get_recording_snapshot(self, session_id: str) -> Dict[str, Any]:
        stats = self._ensure_recording_stats(session_id)
        return {
            "recorded_ms": self._get_recorded_ms(session_id),
            "recording": bool(stats.get("recording")),
        }

    @staticmethod
    def _compose_transcript(items: List[Dict[str, Any]]) -> str:
        parts = [
            item.get("text", "").strip()
            for item in items
            if item.get("text")
        ]
        return " ".join(parts).strip()

    def _on_event(self, event: ASREvent, source: str = "mic"):
        """Handle ASR event and queue for broadcast"""
        if self._loop:
            self._loop.call_soon_threadsafe(
                lambda e=event, s=source: asyncio.create_task(self._handle_event(e, s))
            )

    async def _handle_event(self, event: ASREvent, source: str = "mic"):
        """Process ASR event and broadcast"""
        if self._is_session_terminated(event.session_id):
            return
        if event.type in {EventType.ASR_FINAL, EventType.ASR_PARTIAL}:
            text = (event.payload.get("text", "") or "").strip()
            if not text:
                return
            event.payload["text"] = text
        # Build event dict with source tagging
        event_dict = event.to_dict()
        if source != "mic":
            event_dict["segment_id"] = f"{source}-{event.segment_id}"
            event_dict["session_id"] = self.engine.session_id  # normalize to primary
        event_dict["source"] = source
        # Broadcast the event
        await self.broadcast(event_dict)

        # On final text, add to transcript and maybe generate insights
        if event.type == EventType.ASR_FINAL:
            text = event.payload.get("text", "")
            if text:
                tagged = f"[{source.upper()}] {text}"
                self.insights.add_text(tagged)
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

        for item in sessions:
            session_id = item.get("session_id")
            if not session_id:
                continue
            if session_id in self._recording_stats:
                item["recorded_ms"] = self._get_recorded_ms(session_id)

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
        if session_id in self._recording_stats:
            data["recorded_ms"] = self._get_recorded_ms(session_id)

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

    def _calc_system_vad_padding(self, mic_padding_ms: int) -> int:
        return max(120, int(mic_padding_ms * 0.35))

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
        source = audio.get("source", self.audio_source)
        system_id = audio.get("systemDevice")

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
                if self.system_engine:
                    self.system_engine.config.vad_end_padding_ms = self._calc_system_vad_padding(padding)
        except (TypeError, ValueError):
            ignored.append("audio.vadSensitivity")

        try:
            sys_max_ms = audio.get("sysMaxSegmentMs", 0)
            if sys_max_ms is None:
                sys_max_ms = 0
            sys_max_ms = int(sys_max_ms)
            if sys_max_ms < 0:
                sys_max_ms = 0
            if self.system_engine:
                self.system_engine.config.max_segment_ms = sys_max_ms or None
            applied.append("audio.sysMaxSegmentMs")
        except (TypeError, ValueError):
            ignored.append("audio.sysMaxSegmentMs")

        if source in {"mic", "system", "both"}:
            if source != self.audio_source:
                self.audio_source = source
                self._reset_aec()
                applied.append("audio.source")
        else:
            ignored.append("audio.source")

        resolved_device = self._resolve_input_device(mic_id)
        if resolved_device != self.audio_device:
            try:
                self.audio_device = resolved_device
                if not self.audio_paused and self.audio_source in {"mic", "both"}:
                    self._start_audio_stream(resolved_device)
                self._reset_aec()
                applied.append("audio.micId")
            except Exception as e:
                ignored.append("audio.micId")
                print(f"{YELLOW}[Audio]{RESET} Failed to switch mic: {e}")

        resolved_output = self._resolve_output_device(system_id)
        if resolved_output != self.system_device:
            try:
                self.system_device = resolved_output
                if not self.audio_paused and self.audio_source in {"system", "both"}:
                    self._start_system_stream(resolved_output)
                self._reset_aec()
                applied.append("audio.systemDevice")
            except Exception as e:
                ignored.append("audio.systemDevice")
                print(f"{YELLOW}[Audio]{RESET} Failed to switch system audio: {e}")

        nlms_cfg = audio.get("nlms", {}) if isinstance(audio, dict) else {}
        if isinstance(nlms_cfg, dict):
            if "enabled" in nlms_cfg:
                self._nlms_enabled = bool(nlms_cfg.get("enabled"))
                applied.append("audio.nlms.enabled")
            if "order" in nlms_cfg:
                try:
                    order = max(64, int(nlms_cfg.get("order", self._nlms_order)))
                    if order != self._nlms_order:
                        self._nlms_order = order
                        self._nlms_w = np.zeros(self._nlms_order, dtype=np.float32)
                    applied.append("audio.nlms.order")
                except Exception:
                    ignored.append("audio.nlms.order")
            if "mu" in nlms_cfg:
                try:
                    self._nlms_mu = float(nlms_cfg.get("mu", self._nlms_mu))
                    applied.append("audio.nlms.mu")
                except Exception:
                    ignored.append("audio.nlms.mu")
            if "eps" in nlms_cfg:
                try:
                    self._nlms_eps = float(nlms_cfg.get("eps", self._nlms_eps))
                    applied.append("audio.nlms.eps")
                except Exception:
                    ignored.append("audio.nlms.eps")

        if not self.audio_paused:
            try:
                self._refresh_audio_streams()
            except Exception as e:
                ignored.append("audio.source")
                print(f"{YELLOW}[Audio]{RESET} Failed to refresh streams: {e}")

        transcription = self.settings.get("transcription", {})
        if "punctuation" in transcription:
            if self.engine:
                self.engine.config.use_punc = bool(transcription.get("punctuation"))
                applied.append("transcription.punctuation")
            if self.system_engine:
                self.system_engine.config.use_punc = bool(transcription.get("punctuation"))
        else:
            ignored.append("transcription.punctuation")
        if "mergeStrategy" in transcription:
            applied.append("transcription.mergeStrategy")
        if "modelMode" in transcription:
            mode = transcription.get("modelMode")
            if self.engine and mode in {"realtime", "offline"}:
                self.engine.set_streaming_enabled(mode == "realtime")
                applied.append("transcription.modelMode")
            if self.system_engine and mode in {"realtime", "offline"}:
                self.system_engine.set_streaming_enabled(mode == "realtime")
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
            applied.append("summary.liveSummary")

        display = self.settings.get("display", {})
        if "autoScroll" in display:
            applied.append("display.autoScroll")
        if "showTimestamps" in display:
            applied.append("display.showTimestamps")
        if "exportFormat" in display:
            applied.append("display.exportFormat")

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

    def _handle_audio_callback(self, indata, frames, time_info, status, source: str):
        """Audio capture callback"""
        if status:
            print(f"{GRAY}Audio: {status}{RESET}", file=sys.stderr)
        if indata is None or len(indata) == 0:
            return
        audio = indata[:, 0].copy()
        if source == "mic":
            if self.audio_gain != 1.0:
                audio = audio * self.audio_gain
                audio = np.clip(audio, -1.0, 1.0)
            if self.noise_gate > 0:
                audio[np.abs(audio) < self.noise_gate] = 0.0
        if source == "system" and self._system_sample_rate != SAMPLE_RATE:
            audio = self._resample_audio(audio, self._system_sample_rate, SAMPLE_RATE)
        target = self._mic_queue if source == "mic" else self._system_queue
        target.put(audio)

    def _audio_callback_mic(self, indata, frames, time_info, status):
        self._handle_audio_callback(indata, frames, time_info, status, "mic")

    def _audio_callback_system(self, indata, frames, time_info, status):
        self._handle_audio_callback(indata, frames, time_info, status, "system")

    @staticmethod
    def _init_com() -> None:
        try:
            ctypes.windll.ole32.CoInitialize(None)
        except Exception:
            pass

    @staticmethod
    def _uninit_com() -> None:
        try:
            ctypes.windll.ole32.CoUninitialize()
        except Exception:
            pass

    @staticmethod
    def _supports_wasapi_loopback() -> bool:
        if not hasattr(sd, "WasapiSettings"):
            return False
        try:
            return "loopback" in inspect.signature(sd.WasapiSettings).parameters
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _block_size_for_rate(samplerate: int) -> int:
        if samplerate <= 0:
            return BLOCK_SIZE
        duration = BLOCK_SIZE / float(SAMPLE_RATE)
        return max(1, int(round(duration * samplerate)))

    def _use_soundcard_loopback(self) -> bool:
        return sc is not None and not self._wasapi_loopback

    def _resolve_loopback_microphone(self, device_index: Optional[int]):
        if sc is None:
            return None
        target_name = ""
        try:
            if device_index is None:
                device_index = sd.default.device[1]
            if device_index is not None:
                info = sd.query_devices(device_index)
                target_name = str(info.get("name", "")).strip()
        except Exception:
            target_name = ""
        try:
            mics = sc.all_microphones(include_loopback=True)
        except Exception as e:
            print(f"{YELLOW}[Audio]{RESET} Loopback list error: {e}")
            return None
        loopbacks = [mic for mic in mics if getattr(mic, "isloopback", False)]
        if not loopbacks:
            return None
        if target_name:
            target_lower = target_name.lower()
            for mic in loopbacks:
                name = str(getattr(mic, "name", "")).lower()
                if target_lower in name or name in target_lower:
                    return mic
        return loopbacks[0]

    def _create_audio_stream(
        self,
        device_index: Optional[int],
        *,
        samplerate: int = SAMPLE_RATE,
        loopback: bool = False,
        callback=None
    ):
        extra_settings = None
        if loopback:
            try:
                if not self._wasapi_loopback:
                    print(f"{YELLOW}[Audio]{RESET} sounddevice loopback not supported; system capture disabled")
                    return None
                extra_settings = sd.WasapiSettings(loopback=True)
            except Exception as e:
                print(f"{YELLOW}[Audio]{RESET} Loopback setup failed: {e}")
                return None
        return sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype='float32',
            blocksize=BLOCK_SIZE,
            device=device_index,
            callback=callback,
            extra_settings=extra_settings
        )

    def _start_audio_stream(self, device_index: Optional[int]):
        if self._audio_stream:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception:
                pass
        self._audio_stream = self._create_audio_stream(
            device_index,
            samplerate=SAMPLE_RATE,
            loopback=False,
            callback=self._audio_callback_mic
        )
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

    def _start_system_stream(self, device_index: Optional[int]):
        if self._system_stream:
            try:
                self._system_stream.stop()
                self._system_stream.close()
            except Exception:
                pass
        if self._system_thread:
            self._system_stop.set()
            self._system_thread.join(timeout=1)
            self._system_thread = None
        samplerate = SAMPLE_RATE
        try:
            if device_index is not None:
                info = sd.query_devices(device_index)
                samplerate = int(info.get("default_samplerate", SAMPLE_RATE))
        except Exception:
            samplerate = SAMPLE_RATE
        self._system_sample_rate = samplerate
        if self._use_soundcard_loopback():
            mic = self._resolve_loopback_microphone(device_index)
            if not mic:
                print(f"{YELLOW}[Audio]{RESET} No loopback mic available for system audio")
                self._system_stream = None
                return
            self._system_stop.clear()
            thread = threading.Thread(
                target=self._soundcard_loopback_worker,
                args=(mic, samplerate),
                daemon=True
            )
            thread.start()
            self._system_thread = thread
            self._system_stream = None
        else:
            self._system_stream = self._create_audio_stream(
                device_index,
                samplerate=samplerate,
                loopback=True,
                callback=self._audio_callback_system
            )
            if self._system_stream:
                self._system_stream.start()
        self.system_device = device_index

    def _stop_system_stream(self):
        if self._system_stream:
            try:
                self._system_stream.stop()
                self._system_stream.close()
            except Exception:
                pass
            self._system_stream = None
        if self._system_thread:
            self._system_stop.set()
            self._system_thread.join(timeout=1)
            self._system_thread = None

    def _soundcard_loopback_worker(self, mic, samplerate: int) -> None:
        frames = self._block_size_for_rate(samplerate)
        self._init_com()
        try:
            with mic.recorder(samplerate=samplerate, channels=1) as recorder:
                while self.running and not self._system_stop.is_set():
                    data = recorder.record(numframes=frames)
                    if data is None or data.size == 0:
                        continue
                    audio = data[:, 0].astype(np.float32, copy=False)
                    if samplerate != SAMPLE_RATE:
                        audio = self._resample_audio(audio, samplerate, SAMPLE_RATE)
                    self._system_queue.put(audio)
        except Exception as e:
            print(f"{YELLOW}[Audio]{RESET} Loopback capture error: {e}")
        finally:
            self._uninit_com()

    @staticmethod
    def _resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate or audio.size == 0:
            return audio
        duration = audio.shape[0] / float(src_rate)
        target_len = max(1, int(round(duration * dst_rate)))
        x_old = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
        x_new = np.linspace(0.0, duration, num=target_len, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32, copy=False)

    @staticmethod
    def _mix_audio(mic: Optional[np.ndarray], system: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mic is None:
            return system
        if system is None:
            return mic
        max_len = max(mic.shape[0], system.shape[0])
        if mic.shape[0] != max_len:
            mic = np.pad(mic, (0, max_len - mic.shape[0]), mode="constant")
        if system.shape[0] != max_len:
            system = np.pad(system, (0, max_len - system.shape[0]), mode="constant")
        mixed = 0.5 * (mic + system)
        return np.clip(mixed, -1.0, 1.0)

    def _reset_aec(self) -> None:
        self._aec_delay_samples = 0
        self._aec_last_update = 0.0
        self._nlms_w = np.zeros(self._nlms_order, dtype=np.float32)

    def _estimate_aec_delay(self, mic: np.ndarray, system: np.ndarray) -> int:
        now = time.time()
        if now - self._aec_last_update < 0.6:
            return self._aec_delay_samples
        ds = 4
        mic_ds = mic[::ds]
        sys_ds = system[::ds]
        if mic_ds.size < 8 or sys_ds.size < 8:
            return self._aec_delay_samples
        max_lag = int((self._aec_max_delay_ms / 1000.0) * (SAMPLE_RATE / ds))
        if max_lag <= 0:
            return self._aec_delay_samples
        corr = np.correlate(mic_ds, sys_ds, mode="full")
        mid = sys_ds.size - 1
        start = max(0, mid - max_lag)
        end = min(corr.size, mid + max_lag + 1)
        segment = corr[start:end]
        idx = int(np.argmax(np.abs(segment)))
        lag = (start + idx) - mid
        denom = (np.linalg.norm(mic_ds) * np.linalg.norm(sys_ds)) + 1e-6
        corr_norm = float(abs(segment[idx]) / denom)
        if corr_norm >= self._aec_min_corr:
            delay = lag * ds
            blended = int(round(0.7 * self._aec_delay_samples + 0.3 * delay))
            self._aec_delay_samples = blended
            self._aec_last_update = now
        return self._aec_delay_samples

    @staticmethod
    def _shift_audio(audio: np.ndarray, delay_samples: int) -> np.ndarray:
        if delay_samples == 0 or audio.size == 0:
            return audio
        if delay_samples > 0:
            delay = min(delay_samples, audio.size)
            padded = np.pad(audio, (delay, 0), mode="constant")
            return padded[:audio.size]
        delay = min(-delay_samples, audio.size)
        padded = np.pad(audio, (0, delay), mode="constant")
        return padded[delay:]

    def _spectral_echo_suppress(self, mic: np.ndarray, system: np.ndarray) -> np.ndarray:
        n = mic.shape[0]
        if n == 0:
            return mic
        size = 1
        while size < n:
            size <<= 1
        mic_pad = np.zeros(size, dtype=np.float32)
        sys_pad = np.zeros(size, dtype=np.float32)
        mic_pad[:n] = mic
        sys_pad[:n] = system
        mic_fft = np.fft.rfft(mic_pad)
        sys_fft = np.fft.rfft(sys_pad)
        power = (np.abs(sys_fft) ** 2)
        gain = self._aec_spec_floor / (power + self._aec_spec_floor)
        clean_fft = mic_fft * gain
        clean = np.fft.irfft(clean_fft, n=size).astype(np.float32, copy=False)
        return clean[:n]

    def _nlms_process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        """Lightweight NLMS for AEC; operates in-place style on a copy."""
        if not self._nlms_enabled:
            return near
        if near.size < self._nlms_order or far.size < self._nlms_order:
            return near
        far = far.astype(np.float32, copy=False)
        near_out = near.astype(np.float32, copy=True)
        w = self._nlms_w
        order = self._nlms_order
        mu = self._nlms_mu
        eps = self._nlms_eps
        for i in range(order - 1, near_out.shape[0]):
            x = far[i - order + 1:i + 1]
            y = float(np.dot(w, x))
            e = near_out[i] - y
            norm = float(np.dot(x, x)) + eps
            w += (mu / norm) * e * x
            near_out[i] = e
        self._nlms_w = w
        return near_out

    def _mix_with_aec(self, mic: Optional[np.ndarray], system: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mic is None:
            return None
        max_len = max(mic.shape[0], system.shape[0]) if system is not None else mic.shape[0]
        if mic.shape[0] != max_len:
            mic = np.pad(mic, (0, max_len - mic.shape[0]), mode="constant")
        sys_rms = float(np.sqrt(np.mean(np.square(system)))) if system is not None and system.size else 0.0
        if system is None or system.shape[0] == 0 or sys_rms < self._aec_min_sys_rms:
            return mic
        if system.shape[0] != max_len:
            system = np.pad(system, (0, max_len - system.shape[0]), mode="constant")
        delay = self._estimate_aec_delay(mic, system)
        sys_aligned = self._shift_audio(system, delay)
        mic_clean = self._spectral_echo_suppress(mic, sys_aligned)
        mic_clean = self._nlms_process(mic_clean, sys_aligned)
        return mic_clean

    def _clear_audio_queue(self):
        try:
            while True:
                self._mic_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            while True:
                self._system_queue.get_nowait()
        except queue.Empty:
            return

    def _refresh_audio_streams(self):
        if self.audio_paused:
            return
        if self.audio_device is None:
            try:
                self.audio_device = sd.default.device[0]
            except Exception:
                self.audio_device = None
        if self.system_device is None:
            try:
                self.system_device = sd.default.device[1]
            except Exception:
                self.system_device = None
        source = self.audio_source
        if source == "mic":
            self._stop_system_stream()
            self._start_audio_stream(self.audio_device)
        elif source == "system":
            self._stop_audio_stream()
            self._start_system_stream(self.system_device)
        else:
            self._start_audio_stream(self.audio_device)
            self._start_system_stream(self.system_device)

    def _set_audio_paused(self, paused: bool):
        self.audio_paused = paused
        if paused:
            self._stop_audio_stream()
            self._stop_system_stream()
            self._clear_audio_queue()
        else:
            self._refresh_audio_streams()

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

    def _resolve_output_device(self, device_id: Optional[str]) -> Optional[int]:
        if device_id is None:
            return None
        device_id = str(device_id).strip()
        if not device_id:
            return None
        try:
            device_index = int(device_id)
            info = sd.query_devices(device_index)
            if info and info.get("max_output_channels", 0) > 0:
                return device_index
        except (ValueError, TypeError):
            device_index = None
        try:
            devices = sd.query_devices()
        except Exception:
            return None
        lower_target = device_id.lower()
        for index, device in enumerate(devices):
            if device.get("max_output_channels", 0) <= 0:
                continue
            name = str(device.get("name", "")).lower()
            if lower_target in name or name in lower_target:
                return index
        return None

    def _list_audio_devices(self) -> Dict[str, List[Dict[str, Any]]]:
        inputs = []
        outputs = []
        try:
            all_devices = sd.query_devices()
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]
        except Exception:
            return {"inputs": inputs, "outputs": outputs}
        for index, device in enumerate(all_devices):
            if device.get("max_input_channels", 0) > 0:
                inputs.append({
                    "id": str(index),
                    "name": device.get("name", f"Input {index}"),
                    "is_default": index == default_input,
                    "default_samplerate": device.get("default_samplerate"),
                })
            if device.get("max_output_channels", 0) > 0:
                outputs.append({
                    "id": str(index),
                    "name": device.get("name", f"Output {index}"),
                    "is_default": index == default_output,
                    "default_samplerate": device.get("default_samplerate"),
                })
        return {"inputs": inputs, "outputs": outputs}

    async def _process_audio(self):
        """Process audio in async loop"""
        while self.running:
            try:
                if not self._engine_ready:
                    await asyncio.sleep(0.05)
                    continue
                if self.audio_paused:
                    self._clear_audio_queue()
                    await asyncio.sleep(0.05)
                    continue
                source = self.audio_source
                mic_chunk = None
                sys_chunk = None
                try:
                    mic_chunk = self._mic_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    sys_chunk = self._system_queue.get_nowait()
                except queue.Empty:
                    pass

                if source == "mic":
                    if mic_chunk is not None:
                        self.engine.feed_audio(mic_chunk)
                elif source == "system":
                    if sys_chunk is not None:
                        if self.system_engine:
                            self.system_engine.feed_audio(sys_chunk)
                else:
                    # dual-channel: mic -> primary ASR, system -> system ASR, no mixing
                    if self.system_engine and sys_chunk is not None:
                        self.system_engine.feed_audio(sys_chunk)
                    if mic_chunk is not None:
                        self.engine.feed_audio(mic_chunk)
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
                "recorded_ms": self._get_recorded_ms(self.engine.session_id),
                "recording": bool(self._ensure_recording_stats(self.engine.session_id).get("recording")),
                "message": "Connected to ASR WebSocket Server with LLM"
            }
        }, ensure_ascii=False))
        await websocket.send(json.dumps({
            "type": "engine.status",
            "payload": {
                "ready": bool(self._engine_ready),
                "error": self._init_error,
            }
        }, ensure_ascii=False))
        await websocket.send(json.dumps({
            "type": "audio.paused",
            "payload": {"paused": self.audio_paused, **self._get_recording_snapshot(self.engine.session_id)}
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
                        self._set_recording_state(old_session_id, False)
                        self.engine.reset()
                        new_session_id = self.engine.new_session()
                        if self.system_engine:
                            self.system_engine.reset()
                            self.system_engine.new_session()
                        self.insights = InsightGenerator(self.llm)
                        self.insights.set_summary_interval(self.settings["summary"]["intervalSec"])
                        self._live_seeded = False
                        self._set_recording_state(new_session_id, not self.audio_paused)
                        if new_session_id in self._terminated_sessions:
                            self._terminated_sessions.pop(new_session_id, None)
                        if self._paused_by_terminate:
                            self._set_audio_paused(False)
                            self._paused_by_terminate = False
                            await self.broadcast({
                                "type": "audio.paused",
                                "payload": {"paused": self.audio_paused, **self._get_recording_snapshot(new_session_id)}
                            })
                        self._set_recording_state(new_session_id, not self.audio_paused)
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
                                "reason": "reset",
                                "recorded_ms": self._get_recorded_ms(new_session_id),
                                "recording": bool(self._ensure_recording_stats(new_session_id).get("recording")),
                            }
                        })
                        await websocket.send(json.dumps({
                            "type": "engine.reset",
                            "payload": {"status": "ok", "session_id": new_session_id}
                        }, ensure_ascii=False))

                    elif cmd == "session.terminate":
                        session_id = data.get("session_id") or self.engine.session_id
                        if session_id:
                            self._set_recording_state(session_id, False)
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
                                    "recorded_ms": self._get_recorded_ms(session_id),
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
                                    "payload": {"paused": self.audio_paused, **self._get_recording_snapshot(self.engine.session_id)}
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
                                "inputs": devices.get("inputs", []),
                                "outputs": devices.get("outputs", []),
                                "devices": devices.get("inputs", []),
                                "selected": str(self.audio_device) if self.audio_device is not None else "",
                                "selected_output": str(self.system_device) if self.system_device is not None else "",
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
                            snapshot = self._get_recording_snapshot(self.engine.session_id)
                            await self.broadcast({
                                "type": "audio.paused",
                                "payload": {"paused": self.audio_paused, **snapshot}
                            })
                            continue
                        try:
                            self._set_audio_paused(bool(requested))
                        except Exception as e:
                            print(f"{YELLOW}[Audio]{RESET} Pause error: {e}")
                        self._set_recording_state(self.engine.session_id, not self.audio_paused)
                        snapshot = self._get_recording_snapshot(self.engine.session_id)
                        await self.broadcast({
                            "type": "audio.paused",
                            "payload": {"paused": self.audio_paused, **snapshot}
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
        mic_config = ASRConfig(
            use_vad=True,
            use_punc=True,
            device="cuda:0",
            final_decode_model=(
                "E:/code/FunASR-main/FunASR-main/models/models/damo/"
                "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
            ),
        )
        system_config = replace(mic_config)
        system_config.vad_end_padding_ms = self._calc_system_vad_padding(mic_config.vad_end_padding_ms)
        sys_max = self.settings.get("audio", {}).get("sysMaxSegmentMs", 0) or 0
        try:
            sys_max = int(sys_max)
        except (TypeError, ValueError):
            sys_max = 0
        if sys_max > 0:
            system_config.max_segment_ms = sys_max
        self.engine = ASREngine(mic_config)
        # System channel ASR (separate engine)
        self.system_engine = ASREngine(system_config)

        # Register event handlers
        for event_type in EventType:
            self.engine.on(event_type, lambda e: self._on_event(e, "mic"))
            self.system_engine.on(event_type, lambda e: self._on_event(e, "system"))

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
            if not self._init_task or self._init_task.done():
                self._init_task = asyncio.create_task(self._initialize_engines())
            # Run audio processing
            await self._process_audio()

    async def cleanup(self):
        """Cleanup resources"""
        await self.llm.close()
        if self.mongo:
            self.mongo.close()
        if self.system_engine:
            self.system_engine.finalize()

    def stop(self):
        """Stop the server"""
        self.running = False
        if self.engine:
            self.engine.finalize()
        if self.system_engine:
            self.system_engine.finalize()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._repair_task:
            self._repair_task.cancel()
        self._stop_audio_stream()
        self._stop_system_stream()


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
