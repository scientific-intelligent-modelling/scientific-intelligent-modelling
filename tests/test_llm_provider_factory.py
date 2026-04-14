import os
import unittest
from unittest import mock

from scientific_intelligent_modelling.algorithms.llmsr_wrapper.llmsr import llm as llmsr_llm
from scientific_intelligent_modelling.algorithms.drsr_wrapper.drsr import llm as drsr_llm
from scientific_intelligent_modelling.srkit import llm as srkit_llm


class LLMProviderFactoryTest(unittest.TestCase):
    def test_llmsr_deepinfra_factory_defaults(self):
        with mock.patch.dict(os.environ, {"DEEPINFRA_API_KEY": "dummy-deepinfra-key"}, clear=False):
            client = llmsr_llm.ClientFactory.from_config(
                {"model": "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct"}
            )
        self.assertIsInstance(client, llmsr_llm.DeepInfraClient)
        self.assertEqual(client.base_url, "https://api.deepinfra.com/v1/openai")
        self.assertEqual(client.model, "meta-llama/Meta-Llama-3.1-8B-Instruct")
        self.assertEqual(client.api_key, "dummy-deepinfra-key")

    def test_drsr_deepinfra_factory_supports_alias(self):
        client = drsr_llm.ClientFactory.from_config(
            {
                "model": "deep-infra/meta-llama/Meta-Llama-3.1-8B-Instruct",
                "api_key": "dummy",
            }
        )
        self.assertIsInstance(client, drsr_llm.DeepInfraClient)
        self.assertEqual(client.base_url, "https://api.deepinfra.com/v1/openai")

    def test_srkit_deepinfra_factory_honors_explicit_base_url(self):
        client = srkit_llm.ClientFactory.from_config(
            {
                "model": "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct",
                "api_key": "dummy",
                "base_url": "https://api.deepinfra.com/v1/openai",
            }
        )
        self.assertIsInstance(client, srkit_llm.DeepInfraClient)
        self.assertEqual(client.base_url, "https://api.deepinfra.com/v1/openai")


if __name__ == "__main__":
    unittest.main()
