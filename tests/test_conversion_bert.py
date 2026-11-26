"""Test BERT classifier conversion with synthetic checkpoints and no model downloads."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.functional import linear
from transformers import BertConfig, BertForSequenceClassification, BertModel

from conversion import ModelBase, ModelType, get_model_architecture, get_model_class
from conversion.base import gguf


class TestBertConversion(unittest.TestCase):
    def make_config(self, num_labels: int = 2) -> BertConfig:
        return BertConfig(
            vocab_size=8,
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=8,
            max_position_embeddings=8,
            num_labels=num_labels,
        )

    def convert(self, checkpoint):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            checkpoint.save_pretrained(path)
            hparams = ModelBase.load_hparams(path, False, guess=False)
            model_class = get_model_class(get_model_architecture(hparams, ModelType.TEXT))
            output = path / "model.gguf"
            converter = model_class(path, gguf.LlamaFileType.ALL_F32, output, hparams=hparams)
            # The test exercises checkpoint weights and metadata without a production tokenizer.
            with patch.object(converter, "set_vocab"):
                converter.write()
            return self.read_gguf(output)

    def read_gguf(self, path):
        reader = gguf.GGUFReader(path)
        tensors = {tensor.name: tensor.data.copy() for tensor in reader.tensors}
        metadata = {name: field.contents() for name, field in reader.fields.items() if name.startswith("bert.classifier.")}
        return tensors, metadata

    def test_sequence_classifier_preserves_pooler_and_logits(self):
        for labels in (["LABEL_0"], ["LABEL_0", "LABEL_1"]):
            with self.subTest(labels=labels):
                config = self.make_config(num_labels=len(labels))
                checkpoint = BertForSequenceClassification(config).eval()
                weights = checkpoint.state_dict()
                weights["bert.pooler.dense.bias"].copy_(torch.tensor([0.5, -0.5, 1.0, -1.0]))
                tensors, metadata = self.convert(checkpoint)

                for gguf_name, hf_name in (
                    ("cls.weight", "bert.pooler.dense.weight"),
                    ("cls.bias", "bert.pooler.dense.bias"),
                    ("cls.output.weight", "classifier.weight"),
                    ("cls.output.bias", "classifier.bias"),
                ):
                    self.assertIn(gguf_name, tensors)
                    np.testing.assert_array_equal(tensors[gguf_name], weights[hf_name].numpy())
                self.assertEqual(metadata["bert.classifier.activation"], "tanh")
                self.assertEqual(metadata["bert.classifier.output_labels"], labels)

                input_ids = torch.tensor([[2, 5, 3], [2, 6, 3]])
                with torch.no_grad():
                    bert_out = checkpoint.bert(input_ids)
                    cls = bert_out[0][:, 0, :] if isinstance(bert_out, tuple) else bert_out.last_hidden_state[:, 0, :]
                    cls_out = checkpoint(input_ids)
                    expected = cls_out[0] if isinstance(cls_out, tuple) else cls_out.logits
                pooler = linear(cls, torch.from_numpy(tensors["cls.weight"]), torch.from_numpy(tensors["cls.bias"]))
                logits = linear(pooler.tanh(), torch.from_numpy(tensors["cls.output.weight"]), torch.from_numpy(tensors["cls.output.bias"]))
                torch.testing.assert_close(logits, expected)

    def test_embedding_model_omits_pooler(self):
        checkpoint = BertModel(self.make_config())
        self.assertIn("pooler.dense.weight", checkpoint.state_dict())
        tensors, metadata = self.convert(checkpoint)
        self.assertIn("token_embd.weight", tensors)
        self.assertFalse(any(name.startswith("cls.") for name in tensors))
        self.assertNotIn("bert.classifier.activation", metadata)

    def test_sequence_classifier_without_pooler(self):
        checkpoint = BertForSequenceClassification(self.make_config())
        setattr(checkpoint.bert, "pooler", None)
        tensors, metadata = self.convert(checkpoint)
        self.assertIn("cls.output.weight", tensors)
        self.assertIn("cls.output.bias", tensors)
        self.assertNotIn("cls.weight", tensors)
        self.assertNotIn("cls.bias", tensors)
        self.assertNotIn("bert.classifier.activation", metadata)


if __name__ == "__main__":
    unittest.main()
