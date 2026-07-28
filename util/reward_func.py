from mathruler.grader import grade_answer
import re
from util.debug import debug


def get_assistant_only(text: str) -> str:
    marker = "<|start_header_id|>assistant<|end_header_id|>"
    if marker in text:
        return text.split(marker, 1)[1]
    return text


def extract_after_think(text: str):
    text = get_assistant_only(text)

    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return None

    if text.index("<think>") > text.index("</think>"):
        return None

    return text.split("</think>", 1)[1]


def extract_boxed_after_think(text: str):
    """
    Strict format extractor:
    only accepts boxed answer after </think>.
    Used for format reward.
    """
    after = extract_after_think(text)
    if after is None:
        return None

    matches = re.findall(r"\\boxed\{([^{}]+)\}", after)
    if not matches:
        return None

    return matches[-1].strip()


def extract_last_boxed_after_open_think(text: str):
    """
    Correctness extractor:
    accepts the last boxed answer anywhere after <think>,
    even if it appears inside the thinking section.
    """
    text = get_assistant_only(text)

    if "<think>" not in text:
        return None

    after_open_think = text.split("<think>", 1)[1]

    matches = re.findall(r"\\boxed\{([^{}]+)\}", after_open_think)
    if not matches:
        return None

    return matches[-1].strip()


def has_correct_format(text: str) -> bool:
    return extract_boxed_after_think(text) is not None


def reward(response: str, ground_truth: str) -> float:
    # For format reward: boxed must be after </think>
    boxed_answer_after_think = extract_boxed_after_think(response)

    # For correctness reward: boxed can be anywhere after <think>
    boxed_answer = extract_last_boxed_after_open_think(response)

    format_reward = 0.5 if boxed_answer_after_think is not None else 0.0

    correct_reward = (
        0.2
        if boxed_answer is not None and grade_answer(boxed_answer, ground_truth)
        else 0.0
    )
    
    pre_correct_reward = (
        1.0
        if boxed_answer_after_think is not None and grade_answer(boxed_answer_after_think, ground_truth)
        else 0.0
    )


    score = format_reward + pre_correct_reward

    debug("*" * 80)
    debug("GT:", ground_truth)
    debug("PRED:", boxed_answer)
    debug("FORMAT_PRED:", boxed_answer_after_think)
    debug("FORMAT:", format_reward)
    debug("CORRECT:", correct_reward)
    debug("SCORE:", score)
    debug("Assistant response:\n", get_assistant_only(response))
    debug("*" * 80)

    return score