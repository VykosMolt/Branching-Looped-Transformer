# Powered clean probe + corrected CoreContent — final verification report

Generated 2026-07-13, Europe/Zagreb. Supersedes the historical orientation-row
split as the headline. Draft v3.16 is **not** modified by this report.

## 1. Powered HH pair-disjoint preference probe

Held-out linear-readout accuracy on `Anthropic/hh-rlhf`, 40,000 source pairs
(32,000 train / 8,000 eval), pair-disjoint split, bias-free L-BFGS logistic
probe on concatenated mean-pooled representations from four post-final-norm
Ouro loop boundaries (feature width 8,192, 384-token truncation, bf16 forward).

| Backbone | Held-out acc. | 95% pair-bootstrap CI |
|---|---:|---|
| Base (`Ouro-2.6B`) | 0.55525 | [0.54438, 0.56625] |
| Thinking (`Ouro-2.6B-Thinking`) | 0.56525 | [0.55425, 0.57600] |
| RLTT (`ouro_rltt_local`) | **0.56975** | [0.55863, 0.58037] |

Integrity checks pass for all three: `exact_antisymmetry = true`,
`max_swap_sum_abs = 0`, 40,000 / 32,000 / 8,000 pairs, no quantization,
no Ouro training or checkpoint writes.

### 1a. Paired comparison (the load-bearing test)

The marginal CIs above **overlap** (Base hi 0.56625 vs RLTT lo 0.55863), which
would naively read as "no difference". That is the wrong test. All three probes
were evaluated on the *identical* 8,000 held-out source pairs by construction
(shared selection seed and split seed), so the comparison is paired, and the
correct interval is a bootstrap over the per-pair difference. It is strictly
tighter, because it removes pair-difficulty variance shared by all backbones.

| Comparison | Δ accuracy | 95% paired CI | p (two-sided) | Significant |
|---|---:|---|---:|---|
| RLTT − Base | +0.0145 | [0.00425, 0.02475] | 0.0046 | **yes** |
| Thinking − Base | +0.0100 | [−0.00013, 0.02013] | 0.0546 | no (borderline) |
| RLTT − Thinking | +0.0045 | [−0.00150, 0.01063] | 0.1604 | no |

With three comparisons, a Bonferroni threshold is α = 0.0167. **RLTT − Base
survives it** (p = 0.0046). The other two do not.

### What this does and does not support

- **Supported:** RLTT carries more linearly-decodable human-preference signal
  than Base. This is the one ordering the data establishes.
- **Not supported:** that RLTT beats Thinking. Δ = +0.0045, p = 0.16 — the
  looped-vs-looped ordering is indistinguishable from noise at this power.
- **Borderline:** Thinking over Base (p = 0.055) is suggestive but fails at 95%.
  Do not report it as an effect.
- Absolute accuracies (~0.55–0.57) are modest. The claim is about the *ordering*
  of a linear readout, not about competitive preference-model performance.

## 2. Corrected Ouro CoreContent (task-disjoint)

| Quantity | Value |
|---|---:|
| Historical stored-split held-out macro top-1 | 0.6691 |
| **Corrected task-disjoint held-out macro top-1** | **0.6310** |
| Difference | −0.0381 |
| Stored task IDs crossing splits | 195 |
| Corrected task IDs crossing splits | 0 |

Corrected split: 23,054 train / 2,948 validation / 2,831 held-out tasks
(28,833 unique). Per-domain held-out top-1: Coding 0.8956 (182), Reasoning
0.5985 (269), Math 0.6409 (298), Logic 0.4057 (212), Alignment 0.6143 (2,564).

The historical 0.6691 does **not** stand unchanged — task leakage across splits
inflated it by ~3.8 points. The signal remains substantial after correction.

## 3. Combined picture

| Result | Value |
|---|---:|
| Ouro CoreContent, corrected task-disjoint | 0.6310 |
| Non-looped MiniCPM, task-disjoint | 0.5680 |
| Ouro HH preference probe, best (RLTT) | 0.5698 |
| Ouro HH preference probe, Base | 0.5553 |

CoreContent's corrected 0.6310 remains well clear of the non-looped MiniCPM
0.5680 baseline. Note these two sit on a different task and split protocol from
the HH probe in §1 and are not directly comparable to it; they are reported
together only as the combined verification picture.

## 4. Provenance

- Extraction/fit: `tools/paper_verification/powered_hh_pair_disjoint_probe.py`
- Paired bootstrap: `tools/paper_verification/paired_bootstrap_powered_hh.py`
- CoreContent refit: `tools/paper_verification/rerun_ouro_corecontent_task_disjoint.py`
- Per-backbone results: `hh_<backbone>_40000_pair_disjoint_results.json`
- Paired comparison: `powered_hh_paired_bootstrap.json`
- CoreContent: `ouro_corecontent_task_disjoint_results.json`

Seeds: pair-selection 20260711, split 20260711, bootstrap 20260711
(10,000 draws throughout).

**Known gap:** the Base backbone has no `hh_base_40000_diffs/manifest.json`.
Its extraction completed in an earlier process; on re-entry `extract` returns
early once existing shards cover all 40,000 pairs, and that return precedes the
manifest write, so the driver will never produce it. Shard count (80) and row
count (40,000) are verified and `fit` validates the row count independently.
Thinking and RLTT have manifests. This is a provenance gap, not a data defect.
