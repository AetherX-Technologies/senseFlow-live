#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict JSON output test for Claude CLI with JSON schema.

No fallback: if structured output is missing or invalid, the script fails.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_SYSTEM_PROMPT = (
    "你是专业会议助手，只能输出 JSON 对象，字段仅限 summary、summary_live、actions、questions，禁止输出多余文字。"
)


DEFAULT_TEMPLATE = (
    "历史摘要（JSON，可能为空）：\n"
    "{previous}\n\n"
    "新增转录内容（本次新增部分）：\n"
    "{transcript}\n\n"
    "任务：\n"
    "- 在历史摘要基础上进行更新：新增内容补充已有条目时要合并扩展，出现冲突时要修正/替换。\n"
    "- 如果历史摘要为空，就直接基于新增转录生成。\n"
    "- 保持重要信息，不要重复；不确定时保留原文表达。\n"
    "- 不要编造内容，所有信息必须来自新增转录。\n\n"
    "输出要求（返回的是更新后的完整摘要）：\n"
    "- 只返回 JSON 对象，不要解释、不要代码块、不要额外文字。\n"
    "- JSON 字段：\n"
    "  1. summary: 3-5 条要点，每条一句话（字符串）。\n"
    "  2. summary_live: 1-3 条要点，只针对“新增转录内容”生成。\n"
    "  3. actions: 待办事项数组，每项包含 text 字段。\n"
    "  4. questions: 悬而未决问题数组，每项包含 text 字段。\n\n"
    "示例：\n"
    "{{\"summary\": [\"要点1\", \"要点2\", \"要点3\"], \"summary_live\": [\"新增要点1\"], \"actions\": [], \"questions\": []}}"
)



def _find_cli_js(cli_js_arg: Optional[str]) -> str:
    if cli_js_arg:
        return cli_js_arg
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidate = Path(appdata) / "npm" / "node_modules" / "@anthropic-ai" / "claude-code" / "cli.js"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("Could not find Claude CLI JS. Use --cli-js to specify it.")


def _load_transcript(path: Optional[str]) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    return sys.stdin.read()


def _load_previous(previous_file: Optional[str], previous_json: Optional[str]) -> Dict[str, Any]:
    if previous_json:
        return json.loads(previous_json)
    if previous_file:
        text = Path(previous_file).read_text(encoding="utf-8")
        return json.loads(text)
    return {"summary": [], "summary_live": [], "actions": [], "questions": []}


def _load_schema(path: Optional[str]) -> str:
    if path:
        schema_path = Path(path)
    else:
        schema_path = Path(__file__).with_name("llm_schema.json")
    return schema_path.read_text(encoding="utf-8-sig")


def _call_claude(cli_js: str, prompt: str, schema: str, model: str, system_prompt: str, timeout: Optional[int]) -> str:
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
        model,
        "--system-prompt",
        system_prompt,
        prompt,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Claude CLI failed.")
    return result.stdout.strip()


def _extract_structured_output(raw_json: str) -> Dict[str, Any]:
    data = json.loads(raw_json)
    if isinstance(data, dict):
        if "structured_output" in data and isinstance(data["structured_output"], dict):
            return data["structured_output"]
        if "result" in data and isinstance(data["result"], str):
            return json.loads(data["result"])
    raise ValueError("Structured output missing or invalid.")


def _build_prompt(template: str, transcript: str, previous: Dict[str, Any]) -> str:
    previous_json = json.dumps(previous, ensure_ascii=False)
    return template.format(
        transcript=transcript,
        previous=previous_json,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict JSON schema test using Claude CLI.")
    parser.add_argument("--cli-js", help="Path to Claude CLI JS (cli.js)")
    parser.add_argument("--schema", help="Path to JSON schema file")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--transcript-file", help="Transcript file (UTF-8)")
    parser.add_argument("--previous-file", help="Previous summary JSON file (UTF-8)")
    parser.add_argument("--previous-json", help="Previous summary JSON string")
    parser.add_argument("--timeout", type=int, default=None)
    args = parser.parse_args()

    transcript = _load_transcript(args.transcript_file).strip()
    if not transcript:
        raise SystemExit("No transcript provided.")

    previous = _load_previous(args.previous_file, args.previous_json)
    cli_js = _find_cli_js(args.cli_js)
    schema = _load_schema(args.schema)
    prompt = _build_prompt(args.template, transcript, previous)

    raw = _call_claude(cli_js, prompt, schema, args.model, args.system, args.timeout)
    structured = _extract_structured_output(raw)
    print(json.dumps(structured, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
