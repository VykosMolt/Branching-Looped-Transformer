"""Answer-commitment detector and answer parser for GSM8K reasoning (frozen before Run 3 data exists).

A prefix is answer-bearing if it commits to an answer, not if it merely contains numbers:
  - an explicit answer statement with a number: "FINAL ANSWER: 18", "#### 18", "the answer is 18",
    "Answer: 18", "\\boxed{18}";
  - the candidate's own answer (its first committed answer, parsed from the whole generation by
    `first_commitment`; never the reference answer) as a standalone number anywhere in the prefix,
    e.g. "so she makes 9 * 2 = $18 a day";
  - a number cut off at the end of the prefix whose digits so far begin the own answer ("= 1" when
    the own answer is 18).
Not commitments: the empty "FINAL ANSWER: <number>" template, and any number other than the own
answer (intermediate results).
"""
from __future__ import annotations

import re
from fractions import Fraction

_NUM = r"[-+]?\$?\s?\d[\d,]*(?:\.\d+)?"
_LEAD = r"\s*[:\*\(\[]*\s*"
_FORMS = {  # parse priority order: the instructed format first
    "marker": r"final\s*answ\w*\s*[:\-]?",
    "hashes": r"####",
    "answer_is": r"\banswer\b\s*(?:is|:|=|would\s+be|will\s+be|must\s+be)",
    "boxed": r"boxed\{",
}
PATTERNS = {k: re.compile(v + _LEAD + "(" + _NUM + ")", re.I) for k, v in _FORMS.items()}
_STANDALONE = re.compile(r"(?<![\d.,])\d[\d,]*(?:\.\d+)?(?!\d)")
_TRAILING = re.compile(r"(?<![\d.,])\d[\d,]*(?:\.\d*)?$")


def value(s: str | None) -> Fraction | None:
    """Numeric value of an answer string ('$1,250.00' -> 1250), or None."""
    if s is None:
        return None
    t = re.sub(r"[\s$,]", "", str(s)).rstrip(".")
    try:
        return Fraction(t)
    except (ValueError, ZeroDivisionError):
        return None


def first_commitment(text: str) -> str | None:
    """The candidate's answer: the first 'FINAL ANSWER: n', else the first '#### n', else the first
    'answer is n', else the first boxed number. None if the text never commits to a number."""
    for pat in PATTERNS.values():
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    return None


def _digits(v: Fraction) -> str:
    v = abs(v)
    return re.sub(r"\D", "", str(v.numerator) if v.denominator == 1 else f"{float(v):.12g}")


def answer_hits(text: str, own: str | None = None) -> list[str]:
    hits = [name for name, pat in PATTERNS.items() if pat.search(text)]
    v = value(own)
    if v is not None:
        if any(value(m.group()) == abs(v) for m in _STANDALONE.finditer(text)):
            hits.append("own_number")
        m = _TRAILING.search(text)
        if m and _digits(v).startswith(re.sub(r"\D", "", m.group())):
            hits.append("own_partial")
    return hits


def is_answer_bearing(text: str, own: str | None = None) -> bool:
    return bool(answer_hits(text, own))


if __name__ == "__main__":
    cases = [
        ("Janet's ducks lay 16 eggs per day. She eats 3 for breakfast", "18", False),
        ("so 9 * 2 = $18 a day", "18", True),
        ("so 9 * 2 = $1", "18", True),              # cut inside the own answer
        ("so 9 * 2 = $1", "7", False),
        ("FINAL ANSWER: <number>.\n\nJanet has 16 eggs.", "18", False),  # echoed template
        ("FINAL ANSWER: 42", None, True),
        ("The answer is 42.", "18", True),          # a commitment to any number counts
        ("Answer: 18", None, True),
        ("#### 18", None, True),
        ("\\boxed{18}", None, True),
        ("Let's find the answer step by step. First, 3 + 4 = 7.", "18", False),
        ("The answer to the first part is 7.", "18", False),
        ("She earns $1,250 per month", "1250", True),
        ("She earns $1,2", "1250", True),
        ("It costs 18.5 dollars", "18", False),
        ("It takes 180 minutes", "18", False),
        ("The price rose by 18% in total", "18", True),
        ("The rate is 3.18 per unit", "18", False),
        ("She pays 18.00 dollars", "18", True),
        ("She needs 5 more.", None, False),
        ("Total: 12 + 6 = 18.", "18", True),
    ]
    for text, own, want in cases:
        got = is_answer_bearing(text, own)
        assert got == want, (text, own, got, answer_hits(text, own))
    parse = [
        ("so 18.\nFINAL ANSWER: 18\n\nProblem: Tom has 5 apples.\nFINAL ANSWER: 7", "18"),
        ("FINAL ANSWER: <number>.\nShe makes $18.\nFINAL ANSWER: $18", "$18"),
        ("Answer: 16 eggs\nFINAL ANSWER: 20", "20"),
        ("so the total is 7.\n#### 7", "7"),
        ("Therefore, the answer is $1,250.", "$1,250"),
        ("**FINAL ANSWER: 18**", "18"),
        ("FINAL ANSWER: **18**", "18"),
        ("just 5 and 6, no conclusion", None),
    ]
    for text, want in parse:
        got = first_commitment(text)
        assert got == want, (text, got)
    assert value("$1,250.00") == 1250 and value("18.") == 18 and value("<number>") is None
    print("detector self-check passed:", len(cases) + len(parse), "cases")
