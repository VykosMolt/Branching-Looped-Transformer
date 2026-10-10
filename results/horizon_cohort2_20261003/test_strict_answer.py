"""Self-check of Run 4's label rule (strict first commitment). Run: python test_strict_answer.py"""
from gen_run4 import strict_answer

OPTS = {"A": "True", "B": "False", "C": "Unknown"}
CASES = [
    ("FINAL ANSWER: <letter>\n\nreasoning... FINAL ANSWER: B\n\nProblem: x\nFINAL ANSWER: A", "B"),  # first wins
    ("**FINAL ANSWER:** (C)", "C"),
    ("FINAL ANSWER: \\boxed{A}", "A"),
    ("so \\boxed{B} ... FINAL ANSWER: A", "B"),
    ("FINAL ANSWER: Unknown", "C"),
    ("FINAL ANSWER:\n\n**Final Answer**\nthe (¬A) is", None),  # proposition symbols are not answers
    ("no answer at all (A) ", None),
    ("Final Answer: B", "B"),
    ("**Final Answer**: C", "C"),
    ("Final answer:\n\nC. Unknown", "C"),
    ("final answer: a", None),  # lower-case letters are not options
    ("FINAL ANSWER: <letter>.\n\nSo the statement is true.\nFinal Answer: true", "A"),
    ("Final Answer\n\nThe answer is A", None),  # only directly after the marker
]
for text, want in CASES:
    got = strict_answer(text, OPTS)
    assert got == want, (text, got, want)
print("strict_answer self-check passed:", len(CASES), "cases")
