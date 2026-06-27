#!/usr/bin/env python3
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length else b''
        print(f"Received POST {self.path}")
        print("Headers:")
        for k, v in self.headers.items():
            print(f"{k}: {v}")
        print("Body:")
        try:
            print(body.decode('utf-8'))
        except Exception:
            print(body)
        sys.stdout.flush()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'OK')


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    srv = HTTPServer(('127.0.0.1', port), Handler)
    print(f"Listening on http://127.0.0.1:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
