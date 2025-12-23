# -*- coding: utf-8 -*-
"""
LLM Client for AI-powered insights
Uses OpenAI-compatible API
"""

import json
import asyncio
import os
import sys
import subprocess
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    import httpx
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx


@dataclass
class LLMConfig:
    """LLM configuration"""
    base_url: str = "http://127.0.0.1:8040/v1"
    api_key: str = "46831818513nn!K"
    model: str = "claude-haiku-4-5-20251001"
    timeout: float = 180.0  # 3 minutes
    max_tokens: int = 1024
    use_outlines: bool = False
    strict_json: bool = True
    use_claude_cli: bool = True
    claude_cli_js: Optional[str] = None
    cli_schema_path: Optional[str] = None
    cli_timeout: Optional[int] = 120


class LLMClient:
    """Async LLM client for generating insights"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._outlines_client = None
        self._outlines_model = None
        self._outlines_chat_cls = None
        self._outlines_available: Optional[bool] = None
        self._outlines_logged = False

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=self.config.timeout
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
        await self._close_outlines_client()

    def _ensure_outlines_path(self) -> None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        outlines_root = os.path.join(root_dir, "third_party", "outlines")
        if os.path.isdir(outlines_root) and outlines_root not in sys.path:
            sys.path.insert(0, outlines_root)

    async def _get_outlines_model(self):
        if not self.config.use_outlines:
            return None
        if self._outlines_available is False:
            return None
        if self._outlines_model is not None:
            return self._outlines_model

        self._ensure_outlines_path()
        try:
            import outlines
            from outlines.inputs import Chat
            import openai
        except Exception as e:
            self._outlines_available = False
            print(f"[LLM] Outlines unavailable: {e}")
            return None

        try:
            self._outlines_client = openai.AsyncOpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            self._outlines_model = outlines.from_openai(self._outlines_client, self.config.model)
            self._outlines_chat_cls = Chat
            self._outlines_available = True
            if not self._outlines_logged:
                print("[LLM] Outlines enabled for JSON output.")
                self._outlines_logged = True
            return self._outlines_model
        except Exception as e:
            self._outlines_available = False
            self._outlines_client = None
            self._outlines_model = None
            print(f"[LLM] Outlines init error ({type(e).__name__}): {e}")
            return None

    async def _close_outlines_client(self) -> None:
        if not self._outlines_client:
            return
        close_fn = getattr(self._outlines_client, "aclose", None) or getattr(self._outlines_client, "close", None)
        if close_fn:
            result = close_fn()
            if asyncio.iscoroutine(result):
                await result
        self._outlines_client = None
        self._outlines_model = None
        self._outlines_chat_cls = None

    def _coerce_list(self, value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return []

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        snippet = text[start:end + 1].strip()
        try:
            json.loads(snippet)
            return snippet
        except json.JSONDecodeError:
            return None

    def _normalize_summary_payload(self, payload: Any) -> Optional[Dict[str, Any]]:
        if payload is None:
            return None
        if hasattr(payload, "model_dump"):
            data = payload.model_dump()
        elif hasattr(payload, "dict"):
            data = payload.dict()
        elif isinstance(payload, dict):
            data = payload
        elif isinstance(payload, str):
            text = payload.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                extracted = self._extract_json_from_text(text)
                if not extracted:
                    return None
                try:
                    data = json.loads(extracted)
                except json.JSONDecodeError:
                    return None
        else:
            return None

        return {
            "summary": self._coerce_list(data.get("summary", [])),
            "summary_live": self._coerce_list(data.get("summary_live", [])),
            "actions": self._coerce_list(data.get("actions", [])),
            "questions": self._coerce_list(data.get("questions", []))
        }

    def _find_claude_cli_js(self) -> str:
        if self.config.claude_cli_js:
            return self.config.claude_cli_js
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidate = os.path.join(
                appdata,
                "npm",
                "node_modules",
                "@anthropic-ai",
                "claude-code",
                "cli.js"
            )
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError("Claude CLI JS not found. Set LLMConfig.claude_cli_js.")

    def _load_cli_schema(self) -> str:
        if self.config.cli_schema_path:
            schema_path = self.config.cli_schema_path
        else:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            schema_path = os.path.join(root_dir, "prototype", "tools", "llm_schema.json")
        with open(schema_path, "r", encoding="utf-8-sig") as handle:
            return handle.read()

    def _parse_cli_structured_output(self, raw: str) -> Optional[Dict[str, Any]]:
        data = json.loads(raw)
        if isinstance(data, dict):
            structured = data.get("structured_output")
            if isinstance(structured, dict):
                return structured
            result = data.get("result")
            if isinstance(result, str):
                return json.loads(result)
        return None

    async def _generate_summary_with_claude_cli(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        cli_js = self._find_claude_cli_js()
        schema = self._load_cli_schema()

        cmd = [
            "node",
            cli_js,
            "-p",
            "--no-session-persistence",
            "--output-format",
            "json",
            "--json-schema",
            schema,
            "--model",
            self.config.model,
            "--system-prompt",
            system_prompt,
            prompt,
        ]

        def _run() -> str:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.config.cli_timeout,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "Claude CLI failed.")
            return result.stdout.strip()

        raw = await asyncio.to_thread(_run)
        structured = self._parse_cli_structured_output(raw)
        normalized = self._normalize_summary_payload(structured)
        if normalized is None:
            raise ValueError(f"[LLM] Claude CLI returned invalid JSON: {raw[:200]}")
        return normalized

    async def _generate_summary_with_outlines(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        model = await self._get_outlines_model()
        if model is None or self._outlines_chat_cls is None:
            return None

        try:
            schema_type = dict
            try:
                from pydantic import BaseModel, Field

                class ActionItem(BaseModel):
                    text: str = Field(..., description="action item text")

                class QuestionItem(BaseModel):
                    text: str = Field(..., description="open question text")

                class SummarySchema(BaseModel):
                    summary: List[str] = Field(default_factory=list)
                    summary_live: List[str] = Field(default_factory=list)
                    actions: List[ActionItem] = Field(default_factory=list)
                    questions: List[QuestionItem] = Field(default_factory=list)

                schema_type = SummarySchema
            except Exception:
                schema_type = dict

            chat = self._outlines_chat_cls(messages)
            result = await model(
                chat,
                schema_type,
                temperature=0.2,
                max_tokens=self.config.max_tokens
            )
            normalized = self._normalize_summary_payload(result)
            if normalized is None:
                preview = result if isinstance(result, str) else str(result)
                print(f"[LLM] Outlines non-JSON: {preview[:160]}")
            return normalized
        except Exception as e:
            print(f"[LLM] Outlines error ({type(e).__name__}): {e}")
            return None

    async def chat(self, messages: List[Dict[str, str]],
                   temperature: float = 0.3,
                   response_format: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Send chat completion request"""
        try:
            client = await self._get_client()
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": self.config.max_tokens
            }
            if response_format:
                payload["response_format"] = response_format
            response = await client.post(
                "/chat/completions",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.ConnectError as e:
            print(f"[LLM] Connection error - is LLM server running at {self.config.base_url}? {e}")
            return None
        except httpx.HTTPStatusError as e:
            print(f"[LLM] HTTP error {e.response.status_code}: {e.response.text[:200]}")
            return None
        except Exception as e:
            print(f"[LLM] Error ({type(e).__name__}): {e}")
            return None

    async def generate_summary(self, transcript: str, previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate meeting summary from transcript"""
        if len(transcript) < 50:
            return {"summary": [], "summary_live": [], "actions": [], "questions": []}

        previous_payload = self._normalize_summary_payload(previous)
        if previous_payload is None:
            previous_payload = {"summary": [], "summary_live": [], "actions": [], "questions": []}
        previous_json = json.dumps(
            {
                "summary": previous_payload.get("summary", []),
                "actions": previous_payload.get("actions", []),
                "questions": previous_payload.get("questions", [])
            },
            ensure_ascii=False
        )

        prompt = f"""历史摘要（JSON，可能为空）：
{previous_json}

新增转录内容（本次新增部分）：
{transcript}

任务：
- 在历史摘要基础上进行更新：新增内容补充已有条目时要合并扩展，出现冲突时要修正/替换。
- 如果历史摘要为空，就直接基于新增转录生成。
- 保持重要信息，不要重复；不确定时保留原文表达。
- 不要编造内容，所有信息必须来自新增转录。

输出要求（返回的是更新后的完整摘要）：
- 只返回 JSON 对象，不要解释、不要代码块、不要额外文字。
- JSON 字段：
  1. summary: 3-5 条要点，每条一句话（字符串）。
  2. summary_live: 1-3 条要点，只针对“新增转录内容”生成。
  3. actions: 待办事项数组，每项包含 text 字段。
  4. questions: 悬而未决问题数组，每项包含 text 字段。

示例：
{{"summary": ["要点1", "要点2", "要点3"], "summary_live": ["新增要点1"], "actions": [], "questions": []}}"""

        system_prompt = "你是专业会议助手，只能输出 JSON 对象，字段仅限 summary、summary_live、actions、questions，禁止输出多余文字。"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        if self.config.use_claude_cli:
            return await self._generate_summary_with_claude_cli(prompt, system_prompt)

        outlines_result = await self._generate_summary_with_outlines(messages)
        if outlines_result is not None:
            return outlines_result
        if self.config.use_outlines and self.config.strict_json:
            raise ValueError("[LLM] Outlines did not return valid JSON.")

        result = await self.chat(
            messages,
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        if result:
            parsed = self._normalize_summary_payload(result)
            if parsed is not None:
                return parsed
            if self.config.strict_json:
                raise ValueError(f"[LLM] Failed to parse JSON: {result[:200]}")
            print(f"[LLM] Failed to parse JSON: {result[:100]}")
        elif self.config.strict_json:
            raise ValueError("[LLM] Empty response from LLM.")

        return {"summary": [], "summary_live": [], "actions": [], "questions": []}

    async def answer_question(self, question: str, transcript: str) -> str:
        """Answer a question based on transcript context"""
        if len(transcript) < 20:
            return "目前转录内容较少，请稍后再问。"

        prompt = f"""基于以下会议转录内容回答问题。

转录内容：
{transcript}

问题：{question}

请简洁回答，如果转录中没有相关信息，请说明。"""

        messages = [
            {"role": "system", "content": "你是一个会议助手，根据会议转录内容回答问题。回答要简洁准确，基于转录内容。"},
            {"role": "user", "content": prompt}
        ]

        result = await self.chat(messages, temperature=0.3)
        return result or "抱歉，无法生成回答。"


class InsightGenerator:
    """Generates insights from accumulated transcript"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self._transcript_parts: List[str] = []
        self._last_summary_length = 0
        self._summary_interval = 500  # chars between summaries
        self._last_summary_index = 0
        self._last_insights: Dict[str, Any] = {"summary": [], "summary_live": [], "actions": [], "questions": []}

    def add_text(self, text: str):
        """Add finalized text to transcript"""
        if text:
            self._transcript_parts.append(text)

    @property
    def full_transcript(self) -> str:
        return " ".join(self._transcript_parts)

    def should_generate_summary(self) -> bool:
        """Check if we should generate a new summary"""
        current_length = len(self.full_transcript)
        return (current_length - self._last_summary_length) >= self._summary_interval

    async def generate_insights(self) -> Optional[Dict[str, Any]]:
        """Generate insights if enough new content"""
        if not self.should_generate_summary():
            return None

        transcript = self.full_transcript
        new_parts = self._transcript_parts[self._last_summary_index:]
        new_text = " ".join(new_parts).strip()
        if not new_text:
            return None

        result = await self.llm.generate_summary(new_text, previous=self._last_insights)
        self._last_summary_length = len(transcript)
        self._last_summary_index = len(self._transcript_parts)
        self._last_insights = result
        return result

    async def answer(self, question: str) -> str:
        """Answer a question about the transcript"""
        return await self.llm.answer_question(question, self.full_transcript)

    def reset(self):
        """Reset transcript state"""
        self._transcript_parts = []
        self._last_summary_length = 0
        self._last_summary_index = 0
        self._last_insights = {"summary": [], "summary_live": [], "actions": [], "questions": []}
