import unittest

from util.opsd import change_prompts


class ChangePromptsTest(unittest.TestCase):
    def test_changes_llama_user_prompt(self):
        message = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Be helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "What is 2 + 2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        changed = change_prompts([message], ["2 + 2 = 4."])[0]

        self.assertIn("What is 2 + 2?\n\nReference Solution:\n2 + 2 = 4.", changed)
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>", changed)

    def test_changes_qwen_user_prompt(self):
        message = (
            "<|im_start|>system\nBe helpful.<|im_end|>\n"
            "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        changed = change_prompts([message], ["2 + 2 = 4."])[0]

        self.assertIn("What is 2 + 2?\n\nReference Solution:\n2 + 2 = 4.", changed)
        self.assertTrue(changed.endswith("<|im_start|>assistant\n"))
        self.assertEqual(changed.count("<|im_start|>user"), 1)


if __name__ == "__main__":
    unittest.main()
