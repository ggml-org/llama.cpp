#!/usr/bin/env python3
# Фейковый llama-server для ручных hardening-тестов gateway (infcore/tests/manual).
# Парсит подмножество аргументов реального llama-server, отдаёт /health и
# OpenAI-эндпоинты. Опции для проверки супервайзера:
#   --ready-delay SEC   : /health отдаёт 503 первые SEC секунд ("модель грузится")
#   --ignore-sigterm    : игнорировать SIGTERM (форсит путь SIGKILL в супервайзере)
import argparse, json, sys, time, signal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

p = argparse.ArgumentParser()
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, required=True)
p.add_argument("--model"); p.add_argument("--ctx-size")
p.add_argument("--n-gpu-layers"); p.add_argument("--api-key")
p.add_argument("--embedding", action="store_true"); p.add_argument("--mmproj")
p.add_argument("--ready-delay", type=float, default=0.0)
p.add_argument("--ignore-sigterm", action="store_true")
args, _ = p.parse_known_args()

if args.ignore_sigterm:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
ready_at = time.time() + args.ready_delay


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path == "/health":
            if time.time() < ready_at:
                self.send_response(503); self.end_headers(); self.wfile.write(b"loading")
            else:
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.end_headers(); self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if body.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for tok in ["Привет", ", ", "мир"]:
                chunk = {"choices": [{"delta": {"content": tok}}]}
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        else:
            resp = {"id": "chatcmpl-fake", "object": "chat.completion",
                    "model": body.get("model", "?"),
                    "choices": [{"message": {"role": "assistant",
                                             "content": "pong from " + str(args.port)}}]}
            out = json.dumps(resp, ensure_ascii=False).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.end_headers(); self.wfile.write(out)


sys.stderr.write(f"[fake-llama] up on {args.host}:{args.port} model={args.model}\n")
sys.stderr.flush()
ThreadingHTTPServer((args.host, args.port), H).serve_forever()
