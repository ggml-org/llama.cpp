#!/usr/bin/env python3
"""Example client for the llama-server /v1/realtime websocket, with the openai SDK.

Start a server with an audio model, e.g. LFM3-Audio:

    llama-server -hf LiquidAI/LFM3-Audio-2B-GGUF

Spoken conversation, the audio of each wav file is one user turn. It is sent in chunks as it would come from a
microphone, the server prefills it as it arrives so the response starts right after the last chunk:

    python examples/realtime_client.py --instructions "Respond with interleaved text and audio." question.wav

Output modalities: left out, the model picks; "text", "audio" or "text,audio" (interleaved) force the response.
Without --instructions, a forced mode comes with the system prompt of the LFM3-Audio or LFM3-Omni model card:

    python examples/realtime_client.py --output-modalities text,audio --out answer.wav question.wav

Tool calls: --tools gives the conversation a demo tool, the client runs its calls and sends the results back:

    python examples/realtime_client.py --tools --text "Set the volume to 30."

Text turns go in with --text, transcription sessions transcribe each file:

    python examples/realtime_client.py --text "What is the capital of France?"
    python examples/realtime_client.py --type transcription --language en speech.wav

Live transcription models (Voxtral Realtime, LFM3-Audio live ASR) transcribe while the audio is sent, a transcription
session opens with ?intent=transcription:

    python examples/realtime_client.py --type transcription speech.wav

Requirements: pip install "openai[realtime]" numpy
"""

import argparse
import asyncio
import base64
import json
import time
import wave

import numpy as np
from openai import AsyncOpenAI


# system prompts of the LiquidAI/LFM3-Audio-2B and LiquidAI/LFM3-Omni-2B model cards for a forced output mode
MODE_INSTRUCTIONS = {
    "audio": {"text,audio": "Respond with interleaved text and audio."},
    "omni":  {"text": "Respond in text.", "text,audio": "Respond in interleaved text and audio."},
}
# and with tools: calls, or an answer in interleaved text and speech
TOOLS_INSTRUCTIONS = "Respond with function calls, and interleaved text and audio.\n\nList of tools:"

# the demo tool of --tools
DEMO_TOOLS = [{
    "type": "function",
    "name": "set_volume",
    "description": "Set the device speaker volume.",
    "parameters": {
        "type": "object",
        "properties": {"volume": {"type": "integer", "description": "Volume from 0 to 100."}},
        "required": ["volume"],
    },
}]


def run_tool(name, arguments):
    if name == "set_volume":
        return {"volume": json.loads(arguments)["volume"]}
    return {"error": f"unknown tool {name}"}


def read_wav(path, rate):
    # PCM16 wav -> mono int16 at the given rate
    with wave.open(path, "rb") as f:
        assert f.getsampwidth() == 2, "the example reads PCM16 wav files"
        x = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32)
        x = x.reshape(-1, f.getnchannels()).mean(axis=1)
        sr = f.getframerate()
    if sr != rate:
        n = int(round(len(x) * rate / sr))
        x = np.interp(np.arange(n) * sr / rate, np.arange(len(x)), x)
    return np.clip(x, -32768, 32767).astype(np.int16)


def write_wav(path, pcm, rate):
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())


async def main(args):
    # the SDK derives a wss:// URL from base_url, a plain http server needs ws://
    client = AsyncOpenAI(base_url=args.url, api_key=args.api_key, websocket_base_url=args.url.replace("http", "ws", 1))
    query = {"intent": "transcription"} if args.type == "transcription" else {}
    async with client.realtime.connect(model=args.model, extra_query=query) as conn:
        ev = await conn.recv()
        if ev.type == "error":
            print("error:", ev.error.message)
            return
        # the audio goes in at the rate of the model, PCM16 mono
        session_info = ev.model_dump(warnings=False)["session"]
        audio_fmt = session_info["audio"]
        card = "omni" if "omni" in str(session_info.get("model", "")).lower() else "audio"
        in_rate = audio_fmt["input"]["format"]["rate"]
        out_rate = (audio_fmt.get("output") or {}).get("format", {}).get("rate", 24000)

        session = {"type": args.type}
        if args.type == "realtime":
            instructions = args.instructions
            if instructions is None:
                instructions = TOOLS_INSTRUCTIONS if args.tools else MODE_INSTRUCTIONS[card].get(args.output_modalities)
            if args.tools:
                session["tools"] = DEMO_TOOLS
            if instructions is not None:
                session["instructions"] = instructions
            if args.output_modalities:
                session["output_modalities"] = args.output_modalities.split(",")
        elif args.language:
            session["audio"] = {"input": {"transcription": {"language": args.language}}}
        await conn.session.update(session=session)  # ty: ignore[invalid-argument-type]
        ev = await conn.recv()
        if ev.type == "error":
            print("error:", ev.error.message)
            return

        turns = [("audio", path) for path in args.wav] + [("text", text) for text in args.text]
        for i, (kind, src) in enumerate(turns):
            t_last = 0.0

            # the input is sent while the events are read: live transcription arrives while the audio is sent
            async def send_turn():
                nonlocal t_last
                if kind == "audio":
                    pcm = read_wav(src, in_rate)
                    step = in_rate * args.chunk_ms // 1000
                    for j in range(0, len(pcm), step):
                        chunk = pcm[j:j + step]
                        await conn.input_audio_buffer.append(audio=base64.b64encode(chunk.tobytes()).decode())
                        if args.realtime:
                            await asyncio.sleep(len(chunk) / in_rate)
                    if args.type == "transcription":
                        # "final" ends the input, models that transcribe with a delay then complete the last item
                        await conn.send({"type": "input_audio_buffer.commit", "final": i == len(args.wav) - 1})  # ty: ignore[invalid-argument-type]
                else:
                    await conn.conversation.item.create(item={
                        "type": "message", "role": "user", "content": [{"type": "input_text", "text": src}]})
                if args.type == "realtime":
                    # the audio not committed yet ends the user turn
                    await conn.response.create()
                t_last = time.perf_counter()

            print(f"--- turn {i} ({kind}: {src})")
            sender = asyncio.create_task(send_turn())
            t_first = None
            audio = []
            async for ev in conn:
                if ev.type == "response.output_text.delta" or ev.type == "conversation.item.input_audio_transcription.delta":
                    t_first = t_first or time.perf_counter()
                    print(ev.delta, end="", flush=True)
                elif ev.type == "response.output_audio.delta":
                    t_first = t_first or time.perf_counter()
                    audio.append(np.frombuffer(base64.b64decode(ev.delta), dtype=np.int16))
                elif ev.type == "response.done":
                    # the calls of the response are run, their results go back and the model answers them
                    calls = [o for o in ev.model_dump()["response"]["output"] if o["type"] == "function_call"]
                    for call in calls:
                        result = json.dumps(run_tool(call["name"], call["arguments"]))
                        print(f"\n[call {call['name']}({call['arguments']}) -> {result}]", flush=True)
                        await conn.conversation.item.create(item={
                            "type": "function_call_output", "call_id": call["call_id"], "output": result})
                    if not calls:
                        break
                    await conn.response.create()
                elif ev.type == "conversation.item.input_audio_transcription.completed":
                    break
                elif ev.type == "error":
                    print("\nerror:", ev.error.message)
                    return
            await sender
            print()
            if t_first is not None and t_first > t_last:
                print(f"first output {1000 * (t_first - t_last):.0f} ms after the end of the input")
            elif t_first is not None:
                print(f"output started {1000 * (t_last - t_first):.0f} ms before the end of the input, "
                      f"done {1000 * (time.perf_counter() - t_last):.0f} ms after it")
            if audio:
                pcm_out = np.concatenate(audio)
                path = f"{args.out.removesuffix('.wav')}_{i}.wav" if len(turns) > 1 else args.out
                write_wav(path, pcm_out, out_rate)
                print(f"audio: {len(pcm_out) / out_rate:.2f} s -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("wav", nargs="*", help="PCM16 wav files, one user turn each")
    p.add_argument("--text", action="append", default=[], help="a user turn as text, after the audio turns, repeatable")
    p.add_argument("--type", choices=["realtime", "transcription"], default="realtime", help="session type")
    p.add_argument("--instructions", help="system prompt of the conversation")
    p.add_argument("--tools", action="store_true", help="give the conversation a demo tool (set_volume), its calls are run")
    p.add_argument("--output-modalities", choices=["text", "audio", "text,audio"], help="default: the model picks")
    p.add_argument("--language", help="transcription language, e.g. en")
    p.add_argument("--chunk-ms", type=int, default=100, help="audio chunk size")
    p.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True, help="send the audio in real time")
    p.add_argument("--out", default="response.wav", help="where the audio of the responses goes")
    p.add_argument("--url", default="http://127.0.0.1:9931/v1")
    p.add_argument("--api-key", default="none")
    p.add_argument("--model", default="default", help="model name, used by the router")
    asyncio.run(main(p.parse_args()))
