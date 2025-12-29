import asyncio
import json

import websockets


async def main():
    uri = "ws://127.0.0.1:8766"
    print("connect", uri)
    async with websockets.connect(uri) as ws:
        start = asyncio.get_event_loop().time()
        while True:
            if asyncio.get_event_loop().time() - start > 10:
                print("done")
                return
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            data = json.loads(msg)
            t = data.get("type")
            src = data.get("source") or data.get("payload", {}).get("source")
            payload = data.get("payload", {})
            text = payload.get("text", "")
            if t in {"asr.partial", "asr.final"}:
                print(f"{t} src={src} text_len={len(str(text))} text={text!r}")


if __name__ == "__main__":
    asyncio.run(main())
