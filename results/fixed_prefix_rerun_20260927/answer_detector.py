"""Answer-commitment detector for Horizon Logic reasoning prefixes (frozen before Run 1 data exists).

A prefix is answer-bearing if it commits to an answer, not if it merely mentions an option:
the final-answer marker followed by an option, an explicit 'answer/option/choice is X', a copula
conclusion ('is True', 'is not false', 'is unknown'), an 'unknown' paraphrase ('cannot be
determined'), a 'therefore/thus/so/hence ... True' sentence, a boxed answer, or a line that is
only an option letter or word.

Not commitments: restated option lists ('True, False, or Unknown', 'A. True B. False ...' are
removed before matching), and the empty 'FINAL ANSWER:' template the model often echoes before
it starts reasoning.
"""
import re

_OPT = r"(?:true|false|unknown)\b"
# Two or more option words joined by commas, slashes, 'or'/'and': a restatement, not an answer.
_LIST = re.compile(r"\b(?:true|false|unknown)\b(?:\s*(?:,|/|\bor\b|\band\b)\s*(?:or\s+|and\s+)?\b(?:true|false|unknown)\b)+", re.I)

PATTERNS = {
    "marker": re.compile(r"final\s*answ\w*\s*[:\-]?\s*[\(\[<\"']?\s*(?:(?-i:[ABC])\b|" + _OPT + r")", re.I),
    "answer_is": re.compile(r"\b(?:answer|option|choice)\b\s*(?:is|:|=|would\s+be|will\s+be|must\s+be)?\s*[\(\[\"']?\s*(?:(?-i:[ABC])\b|" + _OPT + r")", re.I),
    "copula": re.compile(r"\b(?:is|be|are|was|remains)\s+(?:not\s+)?(?:definitely\s+|therefore\s+|thus\s+|necessarily\s+|clearly\s+)?" + _OPT, re.I),
    "unknown_paraphrase": re.compile(r"\b(?:cannot|can't|can\s+not|could\s+not|couldn't)\s+(?:be\s+)?(?:determined|known|concluded|inferred|deduced|decided|proven|proved)\b", re.I),
    "therefore": re.compile(r"\b(?:therefore|thus|hence|so|consequently)\b[^.\n]{0,100}\b" + _OPT, re.I),
    "boxed": re.compile(r"boxed\{\s*(?:(?-i:[ABC])|" + _OPT + r")", re.I),
    "bare_line": re.compile(r"^\s*[\(\[]?\s*(?:(?-i:[ABC])|" + _OPT + r")\s*[\)\]]?\s*[\.!]?\s*$", re.I | re.M),
}


_NEG_PREFIX = "it is not the case that "


def statement_forms(statement: str) -> list[str]:
    """The task's statement and its two negated surface forms (lowercase, no final period)."""
    core = statement.lower().strip().rstrip(".").strip()
    forms = {core, _NEG_PREFIX + core}
    if " is " in core:
        forms.add(core.replace(" is ", " is not ", 1))
    if core.startswith(_NEG_PREFIX):
        forms.add(core[len(_NEG_PREFIX):])
    return sorted(f for f in forms if f)


def _statement_patterns(statement: str) -> dict[str, re.Pattern]:
    alt = "|".join(re.escape(f) for f in statement_forms(statement))
    return {
        "final_statement": re.compile(r"final\s*answ\w*\s*[:\-]?[\s\S]{0,30}?\b(?:" + alt + r")\b", re.I),
        "conclude_statement": re.compile(
            r"\b(?:therefore|thus|hence|so|conclude[sd]?\s+that|concluding\s+that|means\s+that|implies\s+that)\b"
            r"[^.\n]{0,60}\b(?:" + alt + r")\b", re.I),
    }


def answer_hits(text: str, statement: str | None = None) -> list[str]:
    text = _LIST.sub(" OPTION_LIST ", text)
    pats = dict(PATTERNS)
    if statement:
        pats.update(_statement_patterns(statement))
    return [name for name, pat in pats.items() if pat.search(text)]


def is_answer_bearing(text: str, statement: str | None = None) -> bool:
    return bool(answer_hits(text, statement))


STATEMENT_RE = re.compile(r"Statement:\s*(.+?)\s*$", re.M)


def task_statement(task_prompt: str) -> str:
    m = STATEMENT_RE.search(task_prompt)
    if m is None:
        raise ValueError("no 'Statement:' line in task prompt")
    return m.group(1)


if __name__ == "__main__":
    cases = {
        "Answer:\n\nThe given facts are:\n1. It is not the case that the engine starts.": False,
        "Therefore, the statement \"the valve is closed\" is True.": True,
        "A. True\nB. False\nC. Unknown": False,
        "D. Unknown\nE. If it is raining, then the engine starts.": False,
        "FINAL ANSW": False,
        "FINAL ANSWER: <letter>": False,
        "FINAL ANSWER: C": True,
        "final answer: true": True,
        "The answer is C.": True,
        "so it cannot be determined": True,
        "\\boxed{C}": True,
        "C": True,
        "Is the following statement True, False, or Unknown?": False,
        "whether the statement \"the fan is running\" is true, false, or unknown": False,
        "I need to determine whether the answer is True, False, or Unknown": False,
        "So the question is whether the statement is True, False, or Unknown.": False,
        "so the statement is True.": True,
        "The statement is not false.": True,
        "Based only on the facts and rules, is the statement true": False,
    }
    for text, want in cases.items():
        got = is_answer_bearing(text)
        assert got == want, (text, got, answer_hits(text))
    st = "the door is locked."
    stmt_cases = {
        "D. The statement is not determined.\nE.\n\nFinal Answer:\n\nThe door is locked.": True,
        "Since the key turned, we can conclude that the door is not locked": True,
        "Therefore it is not the case that the door is locked": True,
        "We need to determine if the statement \"the door is locked\" holds.": False,
        "First, the fact given is that the gate is open.": False,
    }
    for text, want in stmt_cases.items():
        got = is_answer_bearing(text, st)
        assert got == want, (text, got, answer_hits(text, st))
    assert task_statement("Facts: x.\nQuestion: ...?\nStatement: the valve is closed.") == "the valve is closed."
    print("detector self-check passed:", len(cases) + len(stmt_cases), "cases")
