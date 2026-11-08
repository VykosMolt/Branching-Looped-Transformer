# T3 fair-tie top-1 on archived series: original vs fair

Post hoc, 2026-11-08. Commit `b8c95e31198315fae4dc574376009d70265a1695`; archive inputs listed in README.md. Fair rule (T1's `fair_top1`): top1_fair = #tied-at-max-score with reward >= max reward / #tied-at-max; exact equality of the scores the original rule ranks (6-dp saved scores for the xloop rows; integer lengths; see task 5 for unrounded scores). Bootstrap where shown: original `bg_xloop_early_v1_stats.bootstrap`, seed 20260720, 10,000 task-clustered draws, the same draws for original and fair. torch 2.12.0.dev20260407+cu128, 4 threads. All values 4 dp.

## Gates (original first-max rule must reproduce the published numbers first)

| task | gate | mismatches | exact check |
|---|---|---|---|
| 1 C3 reverse transfer | PASS | 0 | saved weights == saved weight_sha; regenerated refit_* (12) and forward transfer_* (18) rows == saved files (canonical JSON); forward agreement stats == train_eval_results.json; forward bootstrap deltas == bootstrap_stats.json; all 36 transfer entries (incl. backward top-1, by-domain, Spearman/Pearson/sign/rank agreement, paired deltas, CIs, p>0) == C3 reverse_transfer_results.json |
| 2 shuffled-label controls | PASS | 0 | refit via train_eval.shuffle_labels + fit_cell_grid (seed 20260720, 36-point grid, val-only selection): heldout_macro_top1, by_domain and chance_macro (4 dp) == train_eval_results.json shuffle_controls; printed selection line (best val, tset, lr, seed) and heldout line == train_eval.log |
| 3 length baseline | PASS | 0 | score = +/- len(candidate_text) (chars), first-max via original group_selection_metrics, domain mean over the 600 heldout groups of subset_groups.jsonl, unweighted macro: macro and per-domain (4 dp) == length_baseline.json, both signs |
| 4 cross-checkpoint base26 | PASS | 0 | original stage_crossmodel (sealed RLTT xloop_taps.pt on this checkpoint's heldout features; write_json captured in memory) == crossmodel_rltt_transfer.json (archive copy, identical to commit copy), every cell and field; feature shards == manifest sha256 |
| 4 cross-checkpoint thinking26 | PASS | 0 | original stage_crossmodel (sealed RLTT xloop_taps.pt on this checkpoint's heldout features; write_json captured in memory) == crossmodel_rltt_transfer.json (archive copy, identical to commit copy), every cell and field; feature shards == manifest sha256 |
| 5 MiniCPM 0.568 | PASS | 0 | stored features (sha256 == manifest) + saved channels (minicpm2_sft_corecontent_v2_task_disjoint.pt) + original split lines and evaluate(): split_audit and blockwise_val / blockwise_heldout (unrounded floats, per domain and macro) == corecontent_control_results_task_disjoint.json |
| 5 Ouro 0.631 | PASS | 0 | CoreContent-v2 feature shards (sha256 == manifest) + saved channels (ouro_corecontent_task_disjoint_policy.pt) + original split (assigned_split) and eval_channels_macro: split_audit, group counts, validation and heldout top1_by_domain and macro (unrounded floats) == ouro_corecontent_task_disjoint_results.json |

## 1. C3 reverse transfer: move minus probe trained at the target loop

Saved `xloop_taps.pt` weights; Spearman = transferred vs target-refit scores on the same candidates (score-based, so the tie rule cannot change it; recomputed values equal C3's).

| move | orig delta | orig CI | excl 0 | fair delta | fair CI | excl 0 | CI conclusion changed | Spearman | ties at max: move / target rows (changed top-1) |
|---|---|---|---|---|---|---|---|---|---|
| 8: 2->1 | -0.0167 | [-0.0550, 0.0200] | no | -0.0167 | [-0.0550, 0.0200] | no | no | 0.7405 | 6 (6) / 6 (6) |
| 8: 3->1 | -0.0667 | [-0.1083, -0.0250] | yes | -0.0667 | [-0.1083, -0.0250] | yes | no | 0.5503 | 6 (6) / 6 (6) |
| 8: 4->1 | -0.0450 | [-0.0850, -0.0033] | yes | -0.0450 | [-0.0850, -0.0033] | yes | no | 0.5963 | 6 (6) / 6 (6) |
| 8: 3->2 | -0.0100 | [-0.0367, 0.0167] | no | -0.0100 | [-0.0367, 0.0167] | no | no | 0.8706 | 7 (7) / 7 (7) |
| 8: 4->2 | +0.0017 | [-0.0200, 0.0250] | no | +0.0017 | [-0.0200, 0.0250] | no | no | 0.9359 | 7 (7) / 7 (7) |
| 8: 4->3 | -0.0150 | [-0.0417, 0.0117] | no | -0.0150 | [-0.0417, 0.0117] | no | no | 0.9099 | 6 (6) / 6 (6) |
| 16: 2->1 | -0.0050 | [-0.0383, 0.0283] | no | -0.0042 | [-0.0375, 0.0300] | no | no | 0.8572 | 6 (6) / 7 (7) |
| 16: 3->1 | -0.0083 | [-0.0433, 0.0283] | no | -0.0075 | [-0.0425, 0.0292] | no | no | 0.8040 | 6 (6) / 7 (7) |
| 16: 4->1 | -0.0217 | [-0.0583, 0.0150] | no | -0.0208 | [-0.0567, 0.0150] | no | no | 0.7895 | 6 (6) / 7 (7) |
| 16: 3->2 | -0.0083 | [-0.0300, 0.0133] | no | -0.0083 | [-0.0300, 0.0133] | no | no | 0.9450 | 6 (6) / 6 (6) |
| 16: 4->2 | -0.0050 | [-0.0300, 0.0200] | no | -0.0050 | [-0.0300, 0.0200] | no | no | 0.9330 | 6 (6) / 6 (6) |
| 16: 4->3 | -0.0050 | [-0.0217, 0.0117] | no | -0.0050 | [-0.0217, 0.0117] | no | no | 0.9804 | 6 (6) / 6 (6) |
| 24: 2->1 | +0.0033 | [-0.0267, 0.0334] | no | +0.0031 | [-0.0267, 0.0333] | no | no | 0.8791 | 6 (6) / 6 (6) |
| 24: 3->1 | +0.0133 | [-0.0233, 0.0500] | no | +0.0131 | [-0.0236, 0.0497] | no | no | 0.7787 | 6 (6) / 6 (6) |
| 24: 4->1 | +0.0100 | [-0.0267, 0.0467] | no | +0.0089 | [-0.0278, 0.0450] | no | no | 0.7671 | 7 (7) / 6 (6) |
| 24: 3->2 | -0.0117 | [-0.0383, 0.0150] | no | -0.0117 | [-0.0383, 0.0150] | no | no | 0.9175 | 5 (5) / 5 (5) |
| 24: 4->2 | -0.0217 | [-0.0500, 0.0067] | no | -0.0228 | [-0.0511, 0.0050] | no | no | 0.8899 | 6 (6) / 5 (5) |
| 24: 4->3 | +0.0083 | [-0.0133, 0.0300] | no | +0.0083 | [-0.0133, 0.0300] | no | no | 0.9586 | 7 (7) / 7 (7) |

CI-excludes-zero conclusions changed: **0 of 18**.

## 2. Shuffled-label controls (refit; weights were never saved)

| cell | orig | fair | fair - orig | selected (val) | ties at max (all-equal / partial; changed) |
|---|---|---|---|---|---|
| 8_L1 | 0.3500 | 0.3465 | -0.0035 | math_logic, lr=0.0003, seed=0 (0.3233) | 3 (2 / 1; 3) |
| 8_L4 | 0.4267 | 0.4208 | -0.0058 | code_math_logic, lr=0.0003, seed=0 (0.3167) | 5 (2 / 3; 5) |
| 16_L2 | 0.4033 | 0.3964 | -0.0069 | math_logic, lr=0.001, seed=0 (0.3100) | 6 (2 / 4; 6) |
| 24_L4 | 0.3367 | 0.3308 | -0.0058 | code_reasoning, lr=0.0003, seed=0 (0.3267) | 5 (2 / 3; 5) |

## 3. Length baseline (score = candidate_text length in characters)

Measures reproducing the published values exactly: chars, utf8_bytes, stripped_chars; fair values identical across them: True.

| rule | orig macro | fair macro | alignment orig -> fair | coding orig -> fair | logic orig -> fair | math orig -> fair | reasoning orig -> fair | ties at max (identical text) | changed top-1 |
|---|---|---|---|---|---|---|---|---|---|
| longest first | 0.3233 | 0.2608 | 0.5083 -> 0.5083 | 0.4917 -> 0.2167 | 0.2833 -> 0.2750 | 0.0250 -> 0.0153 | 0.3083 -> 0.2889 | 117 of 600 (0) | 67 |
| shortest first | 0.3700 | 0.2321 | 0.4917 -> 0.4917 | 0.0417 -> 0.0132 | 0.2417 -> 0.2278 | 0.9000 -> 0.2556 | 0.1750 -> 0.1722 | 152 of 600 (0) | 124 |

Ties at the max by domain (longest / shortest): alignment 0 / 0, coding 53 / 10, logic 9 / 12, math 40 / 113, reasoning 15 / 17.

## 4. Cross-checkpoint transfer: RLTT taps minus each checkpoint's own probe

| cell | base26 orig | base26 fair | thinking26 orig | thinking26 fair | base26 / thinking26 Pearson, Spearman (unchanged) | ties at max base26 tap/local, thinking26 tap/local |
|---|---|---|---|---|---|---|
| 16_L1 | +0.0000 | +0.0000 | +0.0133 | +0.0133 | 0.9876, 0.9853 / 0.9553, 0.9487 | 6/6, 7/7 |
| 16_L2 | -0.0050 | -0.0050 | +0.0050 | +0.0050 | 0.9816, 0.9783 / 0.9998, 0.9995 | 6/6, 6/6 |
| 16_L3 | +0.0033 | +0.0033 | -0.0033 | -0.0033 | 0.9779, 0.9750 / 0.9997, 0.9996 | 7/7, 7/7 |
| 16_L4 | -0.0017 | -0.0017 | +0.0083 | +0.0083 | 0.9509, 0.9457 / 0.9997, 0.9995 | 7/7, 7/7 |
| 24_L1 | -0.0183 | -0.0172 | +0.0000 | -0.0000 | 0.9708, 0.9671 / 0.9966, 0.9961 | 5/6, 6/6 |
| 24_L2 | +0.0017 | +0.0017 | +0.0017 | +0.0017 | 0.9745, 0.9720 / 0.9981, 0.9976 | 6/6, 5/5 |
| 24_L3 | -0.0017 | -0.0017 | +0.0067 | +0.0067 | 0.9632, 0.9606 / 0.9917, 0.9904 | 7/7, 7/7 |
| 24_L4 | -0.0100 | -0.0100 | -0.0083 | -0.0083 | 0.9487, 0.9444 / 0.9949, 0.9944 | 7/7, 7/7 |
| 36_L4 | +0.0167 | +0.0167 | +0.0283 | +0.0283 | 0.9051, 0.8966 / 0.9278, 0.9227 | 7/7, 7/7 |
| 47_L4 | -0.0017 | -0.0017 | +0.0050 | +0.0050 | 0.8977, 0.8746 / 0.9912, 0.9893 | 7/7, 7/7 |
| 8_L1 | +0.0033 | +0.0033 | +0.0000 | +0.0000 | 0.9866, 0.9845 / 1.0000, 1.0000 | 7/7, 6/6 |
| 8_L2 | +0.0050 | +0.0050 | +0.0017 | +0.0017 | 0.9758, 0.9741 / 0.9982, 0.9979 | 7/7, 7/7 |
| 8_L3 | +0.0017 | +0.0017 | +0.0067 | +0.0067 | 0.9868, 0.9853 / 0.9881, 0.9866 | 7/7, 6/6 |
| 8_L4 | -0.0050 | -0.0050 | +0.0167 | +0.0167 | 0.9801, 0.9783 / 0.9387, 0.9354 | 7/7, 7/7 |

- base26: largest |delta| orig -0.0183 (24_L1), fair -0.0172 (24_L1).
- thinking26: largest |delta| orig +0.0283 (36_L4), fair +0.0283 (36_L4).

## 5. MiniCPM non-looped control vs Ouro task-disjoint CoreContent

Fair rule on the unrounded scores the original argmax used (exact equality); `6dp` repeats it on scores rounded to 6 decimals, as T1's saved files were.

| model | orig macro | fair macro | fair macro (6dp) | coding orig -> fair | reasoning orig -> fair | math orig -> fair | logic orig -> fair | alignment orig -> fair | heldout groups | ties at max (all-equal / partial; changed) |
|---|---|---|---|---|---|---|---|---|---|---|
| MiniCPM-2B-sft | 0.5680 | 0.5437 | 0.5425 | 0.9121 -> 0.8906 | 0.5204 -> 0.5112 | 0.3893 -> 0.3866 | 0.3255 -> 0.3255 | 0.6927 -> 0.6045 | 3525 | 467 (461 / 6; 467) |
| Ouro (task-disjoint) | 0.6310 | 0.6265 | 0.6265 | 0.8956 -> 0.8860 | 0.5985 -> 0.5985 | 0.6409 -> 0.6383 | 0.4057 -> 0.4057 | 0.6143 -> 0.6041 | 3525 | 56 (53 / 3; 56) |

MiniCPM sensitivity: in a 2-candidate group each channel's z-scores are exactly (-1, +1), so when the 24_L4 and 36_L4 channels disagree the averaged scores are mathematically (0, 0); float32 noise (up to ~2e-5) breaks some of these ties. Treating all of them as ties: macro **0.5422** (219 groups added to the exact ties; alignment 0.5991). Ouro averages 3 channels (each +/-1), so a sign disagreement alone cannot cancel there; float-broken ties via a zeroed channel were not checked for Ouro.

Ouro minus MiniCPM: orig +0.0630, fair (exact) +0.0829, fair with the MiniCPM sensitivity +0.0843.
