# Hand-audit review (written before any held-out label was opened)

Reviewed 2026-09-27 ~07:10, after the label-free audit stage and before `--stage full`.

All 20 answer-free training prefixes at N* = 48 in `hand_audit.md` were read.

**No model commitment found.** Their content falls into three groups:
- continuations of the option list ("D. ... E. ...", including invented options such as "D. There is not enough information to determine");
- echoes of the empty FINAL ANSWER template ("FINAL ANSWS:", "<answer>", "<step-by-1>");
- the start of the reasoning.

**Two restate a given fact that settles the statement:** #7 (1112) and #8 (482).
- This is prompt content that the probe and the prompt-only baseline both see. It is not a conclusion by the model.
- Not treated as a detector failure. It is noted as a limitation: for tasks whose statement is a given fact, the prompt itself reveals the answer.

The detector is unchanged; its SHA-256 still matches SEAL.sha256.
