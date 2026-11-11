import unittest
from types import SimpleNamespace

from lcb_runner.lm_styles import LMStyle
from lcb_runner.prompts import (
    format_prompt_generation,
    format_prompt_self_repair,
    format_prompt_test_output,
)


class PromptSdkIndependenceTests(unittest.TestCase):
    def setUp(self):
        self.question = SimpleNamespace(
            question_content="Return the product.",
            starter_code="def product(a, b):\n    pass",
            test=[SimpleNamespace(input="2\n3")],
        )

    def prompts(self, style):
        return [
            format_prompt_generation(self.question, style),
            format_prompt_test_output(self.question, style),
            format_prompt_self_repair("Return the product.", style, "pass", [], "{}"),
        ]

    def test_legacy_claude_delimiters_are_preserved(self):
        for prompt in self.prompts(LMStyle.Claude):
            with self.subTest(prompt=prompt[:60]):
                self.assertTrue(prompt.startswith("\n\nHuman:"))
                self.assertTrue(prompt.endswith("\n\nAssistant:"))
                self.assertNotIn("None", prompt)

    def test_other_provider_prompts_import_and_render(self):
        for prompt in self.prompts(LMStyle.OpenAIChat):
            with self.subTest(prompt=prompt):
                self.assertIsInstance(prompt, list)
                self.assertEqual(prompt[0]["role"], "system")
                self.assertEqual(prompt[-1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
