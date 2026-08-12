from mathruler.grader import grade_answer
from util.debug import debug


def get_assistant_only(text: str) -> str:
    marker = "<|start_header_id|>assistant<|end_header_id|>"

    if marker in text:
        return text.split(marker, 1)[1]

    return text


def extract_after_think(text: str):
    """
    Strictly require exactly one <think>...</think> block,
    then return everything after </think>.
    """
    text = get_assistant_only(text)

    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return None

    if text.index("<think>") > text.index("</think>"):
        return None

    return text.split("</think>", 1)[1]


def extract_last_boxed(text: str):
    """
    Extract the last \\boxed{...} expression while correctly
    handling nested braces.

    Examples:
        \\boxed{42}
        \\boxed{\\frac{1}{2}}
        \\boxed{\\dfrac{\\pi}{48}}
        \\boxed{\\sqrt{\\frac{3}{5}}}
    """
    marker = r"\boxed{"

    starts = []
    pos = 0

    while True:
        idx = text.find(marker, pos)

        if idx == -1:
            break

        starts.append(idx)
        pos = idx + len(marker)

    if not starts:
        return None

    # Search from the last boxed expression backward.
    for start in reversed(starts):
        content_start = start + len(marker)

        depth = 1

        for i in range(content_start, len(text)):
            char = text[i]

            if char == "{":
                depth += 1

            elif char == "}":
                depth -= 1

                if depth == 0:
                    return text[content_start:i].strip()

    return None


def extract_boxed_after_think(text: str):
    """
    Strict format extractor.

    Only accepts a boxed answer appearing AFTER </think>.
    Used for format reward and final correctness reward.
    """
    after = extract_after_think(text)

    if after is None:
        return None

    return extract_last_boxed(after)


def extract_last_boxed_after_open_think(text: str):
    """
    Correctness/debug extractor.

    Accepts the last boxed answer anywhere after <think>,
    including inside the thinking section.
    """
    text = get_assistant_only(text)

    if "<think>" not in text:
        return None

    after_open_think = text.split("<think>", 1)[1]

    return extract_last_boxed(after_open_think)


def has_correct_format(text: str) -> bool:
    return extract_boxed_after_think(text) is not None


def reward(response: str, ground_truth: str) -> float:
    # Boxed answer must appear after </think>
    boxed_answer_after_think = extract_boxed_after_think(response)

    # Last boxed answer anywhere after <think>
    # Kept only for debugging / inspection.
    boxed_answer = extract_last_boxed_after_open_think(response)

    # Format reward
    format_reward = (
        0.5
        if boxed_answer_after_think is not None
        else 0.0
    )

    # Final-answer correctness reward
    pre_correct_reward = (
        1.0
        if (
            boxed_answer_after_think is not None
            and grade_answer(
                boxed_answer_after_think,
                ground_truth
            )
        )
        else 0.0
    )

    score = format_reward + pre_correct_reward

    debug("*" * 80)
    debug("GT:", ground_truth)
    debug("PRED:", boxed_answer)
    debug("FORMAT_PRED:", boxed_answer_after_think)
    debug("FORMAT:", format_reward)
    debug("CORRECT:", pre_correct_reward)
    debug("SCORE:", score)
    debug(
        "Assistant response:\n",
        get_assistant_only(response)
    )
    debug("*" * 80)

    return score