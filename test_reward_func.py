import unittest
import sys
from types import ModuleType


# The extraction tests do not exercise grading. Keep them runnable in the
# lightweight test environment, where the training-only dependency is absent.
try:
    import mathruler.grader  # noqa: F401
except ImportError:
    mathruler = ModuleType("mathruler")
    grader = ModuleType("mathruler.grader")
    grader.grade_answer = lambda prediction, answer: prediction == answer
    mathruler.grader = grader
    sys.modules["mathruler"] = mathruler
    sys.modules["mathruler.grader"] = grader

from util.reward_func import get_assistant_only, has_correct_format


class AssistantResponseExtractionTest(unittest.TestCase):
    def test_extracts_llama_instruct_response(self):
        text = (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Question<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "<think>work</think>\\boxed{42}<|eot_id|>"
        )

        self.assertEqual(
            get_assistant_only(text),
            "\n\n<think>work</think>\\boxed{42}<|eot_id|>",
        )
        self.assertTrue(has_correct_format(text))

    def test_extracts_qwen_instruct_response(self):
        text = (
            "<|im_start|>user\nQuestion<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>work</think>\\boxed{42}<|im_end|>"
        )

        self.assertEqual(
            get_assistant_only(text),
            "<think>work</think>\\boxed{42}<|im_end|>",
        )
        self.assertTrue(has_correct_format(text))

    def test_qwen_prompt_think_tags_do_not_affect_response_format(self):
        text = (
            "<|im_start|>system\nUse <think> and </think>.<|im_end|>\n"
            "<|im_start|>user\nQuestion<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>work</think>\\boxed{42}<|im_end|>"
        )

        self.assertTrue(has_correct_format(text))


if __name__ == "__main__":
    unittest.main()
