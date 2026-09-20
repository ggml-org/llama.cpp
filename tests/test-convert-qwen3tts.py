#!/usr/bin/env python3
"""Regression test for https://github.com/ggml-org/llama.cpp/issues/29088.

The Qwen3-TTS CustomVoice variant has no ``speaker_encoder_config`` entry in
its hyperparameters, so ``Qwen3TTSSpeakerEncoderModel.__init__`` raised
``KeyError: 'speaker_encoder_config'`` and ``convert_hf_to_gguf.py --mmproj``
could not convert that variant at all.

Run from the repository root:

    python3 tests/test-convert-qwen3tts.py

Requires torch, numpy and the ``gguf`` package from ``gguf-py/``.
``transformers`` is stubbed out: this test only exercises the hyperparameter
handling in ``__init__`` and never loads weights or configs from disk, so the
real package is not needed.
"""

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))
sys.path.insert(0, str(REPO_ROOT))

# conversion/base.py imports transformers at module level; stub it since this
# test never touches model loading.
_transformers = types.ModuleType("transformers")


class _AutoConfig:  # placeholder, never instantiated here
    pass


_transformers.AutoConfig = _AutoConfig
sys.modules.setdefault("transformers", _transformers)

from conversion import qwen3tts  # noqa: E402


def _run_init(hparams):
    """Run Qwen3TTSSpeakerEncoderModel.__init__ with explicit hparams.

    The real MmprojModel.__init__ loads tensors from disk; stub it out and
    capture the hparams it would have received.
    """
    captured = {}
    orig_init = qwen3tts.MmprojModel.__init__

    def fake_init(self, dir_model, *args, **kwargs):
        captured.update(kwargs)

    qwen3tts.MmprojModel.__init__ = fake_init
    try:
        obj = qwen3tts.Qwen3TTSSpeakerEncoderModel.__new__(
            qwen3tts.Qwen3TTSSpeakerEncoderModel
        )
        qwen3tts.Qwen3TTSSpeakerEncoderModel.__init__(
            obj, Path("/nonexistent"), hparams=dict(hparams)
        )
    finally:
        qwen3tts.MmprojModel.__init__ = orig_init
    return captured.get("hparams")


def test_custom_voice_variant_without_speaker_encoder():
    # Mirrors the CustomVoice variant's config: no speaker_encoder_config key.
    hparams = {"talker_config": {"hidden_size": 1024}}
    out = _run_init(hparams)
    assert "speaker_encoder_config" not in out
    assert out["text_config"] == {"hidden_size": 1024}


def test_base_variant_still_gets_n_layers_patch():
    hparams = {
        "talker_config": {"hidden_size": 1024},
        "speaker_encoder_config": {"hidden_size": 512},
    }
    out = _run_init(hparams)
    assert out["speaker_encoder_config"]["n_layers"] == 4


if __name__ == "__main__":
    test_custom_voice_variant_without_speaker_encoder()
    print("PASS: CustomVoice variant (no speaker_encoder_config) converts")
    test_base_variant_still_gets_n_layers_patch()
    print("PASS: Base variant still gets the n_layers = 4 patch")
    print("All regression tests passed")
