import re
from typing import List, Pattern, Tuple


USER_PATTERNS: Tuple[Pattern[str], ...] = (
    # Llama 3 chat template
    re.compile(
        r"(<\|start_header_id\|>user<\|end_header_id\|>\s*)"
        r"(.*?)"
        r"(\s*<\|eot_id\|>)",
        flags=re.DOTALL,
    ),
    # Qwen 2/2.5/3 chat template
    re.compile(
        r"(<\|im_start\|>user\s*)"
        r"(.*?)"
        r"(\s*<\|im_end\|>)",
        flags=re.DOTALL,
    ),
)


def create_teacher_prompt(
    question: str,
    reference_solution: str,
) -> str:
    return (
        f"{question.strip()}\n\n"
        "Reference Solution:\n"
        f"{reference_solution.strip()}\n\n"
        "Use the verified solution to assess and improve the reasoning process.\n"
        "Continue by producing a correct solution to the original problem."
    )


def change_prompts(
    messages: List[str],
    reference_answers: List[str],
) -> List[str]:
    if len(messages) != len(reference_answers):
        raise ValueError(
            "messages and reference_answers must have the same length."
        )

    changed_messages = []

    for message, reference_answer in zip(messages, reference_answers):
        user_pattern = next(
            (pattern for pattern in USER_PATTERNS if pattern.search(message)),
            None,
        )

        if user_pattern is None:
            raise ValueError(
                "Could not find a Llama or Qwen user section in the message."
            )

        match = user_pattern.search(message)
        # The successful search above guarantees a match for this immutable string.
        assert match is not None

        # Everything between the first user header and its template end token.
        question = match.group(2).strip()

        teacher_prompt = create_teacher_prompt(
            question=question,
            reference_solution=reference_answer,
        )

        changed_message = user_pattern.sub(
            lambda m: (
                f"{m.group(1)}"
                f"{teacher_prompt}"
                f"{m.group(3)}"
            ),
            message,
            count=1,
        )

        changed_messages.append(changed_message)

    return changed_messages
