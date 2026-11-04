"""Exact, automatically verifiable reward for the Option A project.

The professor asked for real numbers and a reproducible answer checker.
The model cannot earn a correctness point just by sounding convincing.
"""
import re

ANSWER_TAG = re.compile(r"<answer>\s*([ABCD])\s*</answer>", re.IGNORECASE)
FORMAT = re.compile(
    r"\s*<think>\s*(?P<explanation>.*?)\s*</think>\s*"
    r"<answer>\s*(?P<letter>[ABCD])\s*</answer>\s*",
    re.IGNORECASE | re.DOTALL,
)


def text_from_completion(value):
    """TRL versions sometimes return text, other times message dictionaries."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks = []
        for part in value:
            if isinstance(part, dict):
                content = part.get("content", "")
                if isinstance(content, list):
                    chunks += [x.get("text", "") for x in content if isinstance(x, dict)]
                elif isinstance(content, str):
                    chunks.append(content)
        return "".join(chunks)
    return ""


def grade_response(response, correct_letter):
    """Correct answer +1.0; usable strict format +0.2; otherwise zero.

    More than one <answer> tag gets ZERO correctness, even if one is right.
    No score for repeated or nonexistent titles; model must select one
    of the four dataset-verified candidates listed in the prompt.
    """
    text = text_from_completion(response)
    found = ANSWER_TAG.findall(text)
    single_answer = len(found) == 1
    predicted = found[0].upper() if single_answer else None
    correct = single_answer and predicted == str(correct_letter).upper()
    format_match = FORMAT.fullmatch(text)
    formatted = bool(format_match and 4 <= len(format_match.group("explanation").split()) <= 100)
    correctness_score = 1.0 if correct else 0.0
    format_score = 0.2 if formatted else 0.0
    return {
        "prediction": predicted,
        "correct": bool(correct), "format_ok": formatted,
        "correctness_reward": correctness_score,
        "format_reward": format_score,
        "reward": correctness_score + format_score,
    }


def grpo_reward(completions, ground_truth, **kwargs):
    """Return one reward per sampled attempt, for TRL's GRPOTrainer."""
    return [grade_response(c, gt)["reward"] for c, gt in zip(completions, ground_truth)]


def run_self_test():
    good = "<think>The listed numbers make option B the best match.</think><answer>B</answer>"
    wrong = "<think>I checked the listed conditions and chose A.</think><answer>A</answer>"
    invalid = "<answer>B</answer><answer>A</answer>"
    assert grade_response(good, "B")["reward"] == 1.2
    assert grade_response(wrong, "B")["reward"] == .2
    assert grade_response(invalid, "B")["reward"] == 0.0
    assert grade_response("<answer>B</answer>", "B")["reward"] == 1.0
    print("Reward tests passed: 1.2, 0.2, 0.0, 1.0")


if __name__ == "__main__":
    run_self_test()
