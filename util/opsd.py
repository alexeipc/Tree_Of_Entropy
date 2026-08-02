import re
from typing import List


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

    user_pattern = re.compile(
        r"(<\|start_header_id\|>user<\|end_header_id\|>\s*)"
        r"(.*?)"
        r"(\s*<\|eot_id\|>)",
        flags=re.DOTALL,
    )

    changed_messages = []

    for message, reference_answer in zip(messages, reference_answers):
        match = user_pattern.search(message)

        if match is None:
            raise ValueError(
                "Could not find a Llama user section in the message."
            )

        # Everything between the first user header and its <|eot_id|>
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