import unittest

from conversion import get_model_class


class ConversionRegistryTest(unittest.TestCase):
    def test_exaone_moe_transformers_alias(self):
        self.assertIs(
            get_model_class("ExaoneMoeForCausalLM"),
            get_model_class("ExaoneMoEForCausalLM"),
        )


if __name__ == "__main__":
    unittest.main()
