"""Post hoc (2026-10-10, Astra audit correction 5): chronological variants of the executed label parsers.

Executed parsers (imported unchanged from their sealed files):
  - Horizon (Run 4 labels; Run 1 strict-label row): gen_run4.strict_answer. A marker+capital letter or a boxed letter,
    whichever comes first; only if neither exists anywhere, the first marker+option word.
  - GSM8K (Run 3 and Run 6 labels): answer_detector_gsm8k.first_commitment (Run 3 dir; Run 6 imports the same file).
    The first 'FINAL ANSWER: n'; only if none, the first '#### n'; then 'answer is n'; then boxed n.
Chronological variants: the same accepted patterns, compiled from the same regex strings; the label is the answer of
the match with the earliest character position (match start). A tie in start position keeps the executed priority.
Run this file to execute the self-test."""
import re
import sys

ROOT = "/home/moloch/ouro_project"
R4 = ROOT + "/opi/preanswer/horizon_cohort2_20261003"
R3 = ROOT + "/opi/preanswer/gsm8k_fixed_prefix_20260927"
for p in (ROOT, ROOT + "/shared", ROOT + "/shared/utilities/tests/manual", R3, R4):
    if p not in sys.path:
        sys.path.insert(0, p)

from gen_run4 import strict_answer  # noqa: E402  executed Horizon parser
import answer_detector_gsm8k as G  # noqa: E402
from answer_detector_gsm8k import first_commitment  # noqa: E402  executed GSM8K parser

# same strings as gen_run4.strict_answer
_LEAD = r"(?i:final\s*answer)[\s\*]*[:\-]?[\s\*\(\[<\"']*"


def _horizon_patterns(options):
    letters = "".join(sorted(options))
    words = {v.strip().rstrip(".").lower(): k for k, v in options.items()}
    return [  # executed priority order, used only to break ties in start position
        (re.compile(_LEAD + rf"([{letters}])\b"), lambda m: m.group(1).upper()),
        (re.compile(rf"\\boxed\{{\s*([{letters}])\s*\}}"), lambda m: m.group(1).upper()),
        (re.compile(_LEAD + r"(" + "|".join(map(re.escape, words)) + r")\b", re.I), lambda m: words[m.group(1).lower()]),
    ]


def strict_answer_chrono(text, options):
    """Horizon: answer of the earliest match among strict_answer's three patterns."""
    hits = []
    for prio, (pat, get) in enumerate(_horizon_patterns(options)):
        m = pat.search(text)
        if m:
            hits.append((m.start(), prio, get(m)))
    return min(hits)[2] if hits else None


def horizon_any_match(text, options):
    """True iff any accepted Horizon pattern matches anywhere in text (chronological scan of a prefix)."""
    return strict_answer_chrono(text, options) is not None


def first_commitment_chrono(text):
    """GSM8K: answer of the earliest match among first_commitment's four patterns (same compiled PATTERNS)."""
    hits = []
    for prio, pat in enumerate(G.PATTERNS.values()):
        m = pat.search(text)
        if m:
            hits.append((m.start(), prio, m.group(1).strip()))
    return min(hits)[2] if hits else None


def _selftest():
    O = {"A": "True", "B": "False", "C": "Unknown"}
    # (text, executed, chronological)
    hz = [
        ("FINAL ANSWER: True ... FINAL ANSWER: B", "B", "A"),  # audit example
        ("**Final Answer**: False\n\nlater: FINAL ANSWER: \\boxed{A}", "A", "B"),
        ("FINAL ANSWER: Unknown. Problem 2 ... \\boxed{A}", "A", "C"),
        ("so \\boxed{B} ... FINAL ANSWER: A", "B", "B"),  # letter vs boxed is already positional
        ("FINAL ANSWER: A ... FINAL ANSWER: False", "A", "A"),
        ("FINAL ANSWER: Unknown", "C", "C"),
        ("Final answer: false ... final answer: true", "B", "B"),
        ("FINAL ANSWER: <letter>\n\nno commitment, (A) and (B)", None, None),
        ("final answer: a", None, None),
    ]
    for t, ex, ch in hz:
        assert strict_answer(t, O) == ex, (t, strict_answer(t, O), ex)
        assert strict_answer_chrono(t, O) == ch, (t, strict_answer_chrono(t, O), ch)
    gs = [
        ("\\boxed{7} ... FINAL ANSWER: 9", "9", "7"),  # audit example
        ("#### 5\n\nFINAL ANSWER: 6", "6", "5"),
        ("Answer: 16 eggs\nFINAL ANSWER: 20", "20", "16"),  # a case in the sealed detector self-test
        ("so the answer is 3. FINAL ANSWER: 4", "4", "3"),
        ("FINAL ANSWER: 18\n\n#### 7", "18", "18"),
        ("the answer is $1,250. \\boxed{3}", "$1,250", "$1,250"),
        ("FINAL ANSWER: <number>.\nShe makes $18.\nFINAL ANSWER: $18", "$18", "$18"),
        ("just 5 and 6, no conclusion", None, None),
    ]
    for t, ex, ch in gs:
        assert first_commitment(t) == ex, (t, first_commitment(t), ex)
        assert first_commitment_chrono(t) == ch, (t, first_commitment_chrono(t), ch)
    # The sealed self-test cases: the chronological parser differs exactly on the priority cases.
    sealed_hz = [  # test_strict_answer.py CASES (text, executed)
        ("FINAL ANSWER: <letter>\n\nreasoning... FINAL ANSWER: B\n\nProblem: x\nFINAL ANSWER: A", "B"),
        ("**FINAL ANSWER:** (C)", "C"), ("FINAL ANSWER: \\boxed{A}", "A"), ("so \\boxed{B} ... FINAL ANSWER: A", "B"),
        ("FINAL ANSWER: Unknown", "C"), ("FINAL ANSWER:\n\n**Final Answer**\nthe (¬A) is", None), ("no answer at all (A) ", None),
        ("Final Answer: B", "B"), ("**Final Answer**: C", "C"), ("Final answer:\n\nC. Unknown", "C"), ("final answer: a", None),
        ("FINAL ANSWER: <letter>.\n\nSo the statement is true.\nFinal Answer: true", "A"), ("Final Answer\n\nThe answer is A", None),
    ]
    diff_hz = [t for t, ex in sealed_hz if strict_answer_chrono(t, O) != ex]
    assert all(strict_answer(t, O) == ex for t, ex in sealed_hz) and diff_hz == [], diff_hz
    sealed_gs = [  # answer_detector_gsm8k.py `parse` cases
        ("so 18.\nFINAL ANSWER: 18\n\nProblem: Tom has 5 apples.\nFINAL ANSWER: 7", "18"),
        ("FINAL ANSWER: <number>.\nShe makes $18.\nFINAL ANSWER: $18", "$18"), ("Answer: 16 eggs\nFINAL ANSWER: 20", "20"),
        ("so the total is 7.\n#### 7", "7"), ("Therefore, the answer is $1,250.", "$1,250"), ("**FINAL ANSWER: 18**", "18"),
        ("FINAL ANSWER: **18**", "18"), ("just 5 and 6, no conclusion", None),
    ]
    diff_gs = [t for t, ex in sealed_gs if first_commitment_chrono(t) != ex]
    assert all(first_commitment(t) == ex for t, ex in sealed_gs) and diff_gs == ["Answer: 16 eggs\nFINAL ANSWER: 20"], diff_gs
    n = len(hz) + len(gs) + len(sealed_hz) + len(sealed_gs)
    print(f"chrono_parsers self-test passed: {n} cases; sealed Horizon cases differing: 0/13; "
          f"sealed GSM8K cases differing: 1/8 ('Answer: 16 eggs ... FINAL ANSWER: 20' -> 16)")


if __name__ == "__main__":
    _selftest()
