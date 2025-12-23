#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe a non-JSON LLM output and normalize it into usable JSON.

This script prefers tagged lines (SUMMARY/ACTION/QUESTION) but will
fallback to JSON extraction or a simple heuristic when the model
ignores formatting.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


DEFAULT_SYSTEM_PROMPT = (
    "You are a meeting assistant. "
    "Extract key points, action items, and open questions. "
    "Follow the output format exactly."
)

DEFAULT_TEMPLATE = (
    "Read the transcript and output ONLY lines with the following prefixes:\n"
    "SUMMARY: <one sentence>\n"
    "ACTION: <one action item>\n"
    "QUESTION: <one open question>\n\n"
    "Rules:\n"
    "- Provide 3-5 SUMMARY lines.\n"
    "- Provide 0+ ACTION lines.\n"
    "- Provide 0+ QUESTION lines.\n"
    "- Do not output any other text.\n\n"
    "Transcript:\n"
    "{transcript}\n"
)

REFORMAT_TEMPLATE = (
    "Rewrite the content into ONLY lines with the following prefixes:\n"
    "SUMMARY: <one sentence>\n"
    "ACTION: <one action item>\n"
    "QUESTION: <one open question>\n\n"
    "Rules:\n"
    "- Provide 3-5 SUMMARY lines.\n"
    "- Provide 0+ ACTION lines.\n"
    "- Provide 0+ QUESTION lines.\n"
    "- Do not output any other text.\n\n"
    "Content:\n"
    "{content}\n"
)

HEADER_PATTERNS = {
    "summary": re.compile(
        r"^(summary|summaries|key points|highlights|\u6458\u8981|\u8981\u70b9|\u603b\u7ed3)\s*[:\uff1a\-]?\s*(.*)$",
        re.IGNORECASE,
    ),
    "actions": re.compile(
        r"^(action|actions|todo|todos|task|tasks|\u5f85\u529e|\u4efb\u52a1|\u884c\u52a8)\s*[:\uff1a\-]?\s*(.*)$",
        re.IGNORECASE,
    ),
    "questions": re.compile(
        r"^(question|questions|open questions|\u95ee\u9898|\u7591\u95ee)\s*[:\uff1a\-]?\s*(.*)$",
        re.IGNORECASE,
    ),
}

BULLET_RE = re.compile(r"^(?:[-*]|\d+[.)])\s*(.+)$")
SENTENCE_SPLIT_RE = re.compile(r"[。！？.!?\n]+")
CLAUSE_SPLIT_RE = re.compile(r"[，,；;、]+")

ACTION_HINTS = (
    "需要",
    "要",
    "请",
    "安排",
    "记录",
    "记一下",
    "完成",
    "准备",
    "提交",
    "提供",
    "确认",
    "收集",
)
QUESTION_HINTS = ("吗", "为什么", "怎么", "如何", "是否", "哪些", "哪", "?", "？")


def _find_cli(cli_arg: Optional[str]) -> str:
    if cli_arg:
        return cli_arg
    env_cli = os.environ.get("CLAUDE_CMD")
    if env_cli:
        return env_cli
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        candidate = os.path.join(user_profile, "AppData", "Roaming", "npm", "claude.cmd")
        if os.path.exists(candidate):
            return candidate
    return "claude"


def _extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    if "```json" in text:
        chunk = text.split("```json", 1)[1].split("```", 1)[0]
        return chunk.strip()
    if "```" in text:
        chunk = text.split("```", 1)[1].split("```", 1)[0]
        return chunk.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1].strip()


def _try_parse_json(text: str) -> Optional[Dict[str, object]]:
    block = _extract_json_block(text)
    if not block:
        return None
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        return None


def _strip_leading_marker(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line)
    return line.strip()


def _match_header(line: str) -> Optional[Tuple[str, str]]:
    cleaned = _strip_leading_marker(line)
    for key, pattern in HEADER_PATTERNS.items():
        match = pattern.match(cleaned)
        if match:
            return key, match.group(2).strip()
    return None


def _parse_tagged_lines(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {"summary": [], "actions": [], "questions": []}
    current: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        header = _match_header(line)
        if header:
            key, content = header
            current = key
            if content:
                sections[key].append(content)
            continue
        bullet = BULLET_RE.match(line)
        if bullet and current:
            sections[current].append(bullet.group(1).strip())
            continue
        if current:
            if sections[current]:
                sections[current][-1] = f"{sections[current][-1]} {line}"
            else:
                sections[current].append(line)
    return sections


def _split_sentences(text: str) -> List[str]:
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if sentences:
        return sentences
    clauses = [c.strip() for c in CLAUSE_SPLIT_RE.split(text) if c.strip()]
    if clauses:
        return clauses
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return []
    step = 40
    return [compact[i:i + step] for i in range(0, len(compact), step)]


def _select_summary(sentences: List[str], max_items: int = 5) -> List[str]:
    filtered = [s for s in sentences if len(s) >= 6]
    if not filtered:
        filtered = sentences
    ranked = sorted(filtered, key=len, reverse=True)
    summary: List[str] = []
    seen = set()
    for sentence in ranked:
        if sentence in seen:
            continue
        summary.append(sentence)
        seen.add(sentence)
        if len(summary) >= max_items:
            break
    if not summary:
        summary = sentences[:max_items]
    return summary


def _extract_actions(sentences: List[str]) -> List[str]:
    actions = []
    for sentence in sentences:
        if any(hint in sentence for hint in ACTION_HINTS):
            actions.append(sentence)
    return actions


def _extract_questions(sentences: List[str]) -> List[str]:
    questions = []
    for sentence in sentences:
        if any(hint in sentence for hint in QUESTION_HINTS):
            questions.append(sentence)
    return questions


def _fallback_heuristic(text: str) -> Dict[str, List[str]]:
    sentences = _split_sentences(text)
    summary = _select_summary(sentences, max_items=5)
    questions = _extract_questions(sentences)
    actions = _extract_actions(sentences)
    return {"summary": summary, "actions": actions, "questions": questions}


def normalize_output(raw_text: str, transcript: Optional[str] = None) -> Dict[str, object]:
    parsed = _try_parse_json(raw_text)
    if isinstance(parsed, dict):
        return {
            "summary": parsed.get("summary", []),
            "actions": parsed.get("actions", []),
            "questions": parsed.get("questions", []),
        }
    sections = _parse_tagged_lines(raw_text)
    if any(sections.values()):
        return {
            "summary": sections["summary"],
            "actions": [{"text": item} for item in sections["actions"]],
            "questions": [{"text": item} for item in sections["questions"]],
        }
    fallback_source = transcript if transcript else raw_text
    fallback = _fallback_heuristic(fallback_source)
    return {
        "summary": fallback["summary"],
        "actions": [{"text": item} for item in fallback["actions"]],
        "questions": [{"text": item} for item in fallback["questions"]],
    }


def run_llm(cli: str, prompt: str, model: str, system_prompt: str, tools: Optional[str], timeout: Optional[int]) -> str:
    cmd = [cli, "-p", prompt, "--model", model, "--system-prompt", system_prompt]
    if tools is not None:
        cmd += ["--tools", tools]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"Command failed: {cmd}")
    return result.stdout.strip()


def load_text(path: Optional[str], inline: Optional[str]) -> str:
    if inline:
        return inline
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    return sys.stdin.read()


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe LLM output and normalize to JSON.")
    parser.add_argument("--cli", help="Path to claude CLI (default: CLAUDE_CMD/env or PATH)")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--reformat-template", default=REFORMAT_TEMPLATE)
    parser.add_argument("--user-prefix", action="store_true", help="Prefix prompt with 'user: ' for Claude CLI")
    parser.add_argument("--retries", type=int, default=1, help="Retries with reformat prompt when output is unusable")
    parser.add_argument("--transcript-file", help="Transcript file (UTF-8)")
    parser.add_argument("--transcript", help="Transcript inline text")
    parser.add_argument("--tools", default="")
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--raw-out", help="Save raw LLM output to a file")
    parser.add_argument("--show-raw", action="store_true")
    parser.add_argument("--parse-only", action="store_true", help="Skip LLM call and parse stdin/raw file")
    args = parser.parse_args()

    if args.parse_only:
        raw = load_text(args.transcript_file, args.transcript)
        transcript = None
    else:
        transcript = load_text(args.transcript_file, args.transcript).strip()
        if not transcript:
            raise SystemExit("No transcript provided.")
        prompt = args.template.format(transcript=transcript)
        if args.user_prefix:
            prompt = f"user: {prompt}"
        cli = _find_cli(args.cli)
        raw = run_llm(cli, prompt, args.model, args.system, args.tools, args.timeout)

        normalized = normalize_output(raw, transcript=transcript)
        has_summary = bool(normalized.get("summary"))
        has_actions = bool(normalized.get("actions"))
        has_questions = bool(normalized.get("questions"))
        retries = args.retries
        while retries > 0 and not (has_summary or has_actions or has_questions):
            re_prompt = args.reformat_template.format(content=raw)
            if args.user_prefix:
                re_prompt = f"user: {re_prompt}"
            raw = run_llm(cli, re_prompt, args.model, args.system, args.tools, args.timeout)
            normalized = normalize_output(raw, transcript=transcript)
            has_summary = bool(normalized.get("summary"))
            has_actions = bool(normalized.get("actions"))
            has_questions = bool(normalized.get("questions"))
            retries -= 1
        result = normalized
        if args.raw_out:
            with open(args.raw_out, "w", encoding="utf-8") as handle:
                handle.write(raw)
        if args.show_raw:
            print(raw, file=sys.stderr)

        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.raw_out:
        with open(args.raw_out, "w", encoding="utf-8") as handle:
            handle.write(raw)
    if args.show_raw:
        print(raw, file=sys.stderr)

    result = normalize_output(raw)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
