# -*- coding: utf-8 -*-
"""Quick LLM connection test"""

import asyncio
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
sys.path.insert(0, 'e:/code/FunASR-main/FunASR-main/prototype')

from asr_engine.llm_client import LLMClient, LLMConfig

async def test():
    print("Testing LLM connection...")
    print(f"  URL: http://127.0.0.1:8040/v1")
    print(f"  Model: claude-haiku-4-5-20251001")
    print()

    config = LLMConfig(
        base_url="http://127.0.0.1:8040/v1",
        api_key="46831818513nn!K",
        model="claude-haiku-4-5-20251001"
    )
    client = LLMClient(config)

    try:
        result = await client.chat([
            {"role": "user", "content": "说一个字：好"}
        ])

        if result:
            print(f"[OK] LLM responded: {result}")
        else:
            print("[FAIL] No response from LLM")

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test())
