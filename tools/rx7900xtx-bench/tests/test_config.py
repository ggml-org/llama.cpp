import unittest

from rx7900xtx_bench.config import resolve_config


class ConfigTest(unittest.TestCase):
    def test_local_values_override_preset(self):
        config = resolve_config("qwen35-iq3s", {"model": "/tmp/model.gguf", "server": "/tmp/llama-server"})
        self.assertEqual(config["context_size"], 200000)
        self.assertEqual(config["model"], "/tmp/model.gguf")
        self.assertIn("--spec-type", config["server_args"])

    def test_local_config_cannot_replace_pinned_server_settings(self):
        with self.assertRaisesRegex(ValueError, "server_args"):
            resolve_config("qwen35-iq3s", {"server_args": ["--spec-type", "none"]})


if __name__ == "__main__":
    unittest.main()
