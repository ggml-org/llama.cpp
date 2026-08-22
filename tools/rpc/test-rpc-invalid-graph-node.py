#!/usr/bin/env python3
import socket
import struct
import subprocess
import sys
import time


def receive_exact(sock, size):
    result = bytearray()
    while len(result) < size:
        chunk = sock.recv(size - len(result))
        if not chunk:
            raise ConnectionError("unexpected EOF")
        result.extend(chunk)
    return bytes(result)


def reserve_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_server(port):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.03)
    raise TimeoutError("RPC server did not bind")


def hello(sock):
    payload = b"\0" * 24
    sock.sendall(struct.pack("<B Q", 14, len(payload)) + payload)
    response_size, = struct.unpack("<Q", receive_exact(sock, 8))
    receive_exact(sock, response_size)


def verify_server_responds(port):
    deadline = time.monotonic() + 2
    last_error = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2) as sock:
                sock.settimeout(0.5)
                hello(sock)
                return
        except (ConnectionError, OSError, TimeoutError) as error:
            last_error = error
            time.sleep(0.03)
    raise AssertionError(f"RPC server stopped responding after invalid graph: {last_error}")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <ggml-rpc-server>")

    port = reserve_port()
    server = subprocess.Popen(
        [sys.argv[1], "-H", "127.0.0.1", "-p", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    error = None
    try:
        wait_for_server(port)
        time.sleep(0.05)
        with socket.create_connection(("127.0.0.1", port), timeout=1) as sock:
            sock.settimeout(1)
            hello(sock)
            graph = struct.pack("<I I Q I", 0, 1, 0, 0)
            sock.sendall(struct.pack("<B Q", 10, len(graph)) + graph)
            if sock.recv(1) != b"":
                raise AssertionError("RPC server kept the invalid graph connection open")
        verify_server_responds(port)
        if server.poll() is not None:
            raise AssertionError(f"RPC server exited after invalid graph: {server.returncode}")
    except Exception as caught:
        error = caught
    finally:
        if server.poll() is None:
            server.terminate()
        try:
            output, _ = server.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()
            output, _ = server.communicate()

    if error is not None:
        raise AssertionError(f"{error}\nRPC server output:\n{output}") from error


if __name__ == "__main__":
    main()
