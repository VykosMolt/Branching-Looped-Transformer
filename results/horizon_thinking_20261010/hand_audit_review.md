# Hand audit review (label-free), 2026-10-18 ~14:05 CEST, before any label was opened

20 answer-free training prefixes at N* = 32 (hand_audit.md, selection as analyze_run8.hand_audit), each read in full with its statement (no labels, no correctness).

**Result: 20 / 20 answer-free.** No prefix commits to an option letter or to a truth value for the statement.

What the prefixes contain:
- Most continue the prompt's option list with invented options ("D. None of the above", "E. The premises are contradictory", empty "F.", "G." ...). Some invented options restate the statement ("F. The light is green", "E. The alarm is on"); these are list items, not assertions.
- Several write an empty answer template ("FINAL ANSWER:", "FINAL ANSW", "FINAL ANSWS:", "Answer:") and echo instructions ("Please reason step by step, and put your final answer within"). The detector treats empty templates as answer-free, as specified.
- A few begin reasoning ("Rationale: The given facts are...", "Reasoning: First, let me recall the facts").

Edge cases (not answers):
- synthetic_propositional::7869 #3 echoes a given fact that settles the statement ("it is not the case that the battery is charged"); this is prompt content, which the probe and the surface features see anyway (as in Run 1's audit).
- synthetic_propositional::7694 #3 writes "Final Answer:" followed by derivation steps ("The alarm is on, so the light is green and it is not raining. The tank is full, so the"), without a conclusion about the statement.

Note for interpretation: at 32 tokens, Ouro-2.6B-Thinking (prompted without a chat template, as in Run 4) is mostly continuing the option list rather than reasoning, so the hidden states at the cut mostly cover invented option text.
