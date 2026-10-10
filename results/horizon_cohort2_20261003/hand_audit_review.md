# Hand audit review (label-free), 2026-10-04 ~11:18, before any label was opened

20 answer-free training prefixes at N* = 48 (hand_audit.md; selection exactly as analyze_run4.hand_audit), each read in full with its statement, the options and the candidate's own strict answer (run4_hand_audit_items.json; no gold answer, no correctness).

**Result: 20 / 20 answer-free.** No prefix commits to an answer: none contains "FINAL ANSWER" (or a variant) followed by an option letter or option word, a boxed letter, or "the answer is ...". The detector missed nothing in this sample.

Notes (not detector misses):
- 12 of 20 open with an empty or truncated answer header before reasoning ("FINAL ANSWER:", "FINAL ANSW", "FINAL ANSWS:", "Final Answer:", one "FINAL ANSWER: <answer>" placeholder). The strict rule correctly does not count these (no capital option letter follows), and neither does the detector.
- 9 of 20 invent extra options ("D. True or False", "D. The signal is sent.", "G. A and E"). These restate or extend the option list and do not commit.
- 3 prefixes restate a given fact that decides the statement: 3563#3 ("The alarm is not on" for "the alarm is on"), 9994#0 ("the engine starts" for "the engine starts"), 9806#2 ("The battery is not charged" for "the battery is charged"). The answer is derivable from the restated prompt, not stated. As on GSM8K, "answer-free" means not yet stated, not underivable; the difficulty features and the end-of-prompt control address this.
- 4 of the 20 sampled answers have no strict label (9581#1, 1401#0, 8372#3, 8494#1) and are excluded from the cohort; the audit sample is over all answer-free training answers.

Audit stage (audit_stage.json): 4,656 records, 1,164 tasks, 0 missing, 0 errors; N* = 48 (answer-free 76.1% of training answers); labelled 70.4%; hit the 448-token cap 36.3%.

Proceeding to `analyze_run4.py --stage full` as sealed (PLAN.md).
