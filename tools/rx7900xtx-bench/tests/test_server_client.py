import unittest

from rx7900xtx_bench.server_client import ServerError, parse_sse_events


class ServerClientTest(unittest.TestCase):
    def test_sse_parser_handles_partial_frames(self):
        events = list(parse_sse_events([
            b'data: {"content":"hel',
            b'lo"}\n\ndata: {"stop":true}\n\n',
            b'data: [DONE]\n\n',
        ]))

        self.assertEqual(events, [{"content": "hello"}, {"stop": True}])

    def test_sse_parser_rejects_invalid_json(self):
        with self.assertRaises(ServerError):
            list(parse_sse_events([b"data: not-json\n\n"]))


if __name__ == "__main__":
    unittest.main()
