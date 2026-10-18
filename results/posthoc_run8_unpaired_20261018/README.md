# Run 8: Thinking-minus-RLTT interval with independent pairing (post hoc, 2026-10-18)

Sealed secondary 7 of Run 8 (`within_oof.unpaired_diff`, from the Run 7 folder) zips the two sides' **sorted** bootstrap
replicates, pairing the k-th smallest Thinking replicate with the k-th smallest RLTT replicate. That comonotone pairing
narrows the interval. Run 8 is the only user of the function (checked on the laptop and the SSD tree). The point
estimate is unaffected. Sealed outputs (`../preanswer/horizon_thinking_20261010/results.json`, `RESULTS.md`) are left as written.

`recompute.py` rebuilds the same out-of-fold scores with Run 8's own sealed code, redraws the same 2,000 replicates per
side in the same RNG order (seed 20260724, one stream, Thinking first), **asserts** that sorting them reproduces the
sealed estimate and interval exactly, and reports the interval from the draws paired in draw order.

| comparison | estimate | sealed (sorted pairing) | independent pairing |
|---|---|---|---|
| own cuts (Thinking 32, RLTT 48) | -0.149 | [-0.191, -0.104] | **[-0.269, -0.023]** |
| both at 48 | -0.143 | [-0.196, -0.082] | **[-0.272, -0.006]** |

Run: `nice -n 10 /home/moloch/ouro_project/shared/venv/bin/python -B recompute.py` (exit 0; `run.log`, `results.json`).
`run_failed_1.log`: first attempt, KeyError 'free' (the script omitted `answer_free_table`); fixed and rerun.
