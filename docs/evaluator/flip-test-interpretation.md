# HH-RLHF Evaluator Flip-Test Interpretation

## Executive summary

The earlier claim that the evaluator’s “real” accuracy might be roughly **65%** was incorrect.

The correct distinction is:

```text
65%:
  the best independent / pointwise evaluator test accuracy

95.2%:
  the pairwise relational evaluator’s fixed-order HH-RLHF test accuracy
```

The low strict sign-flip rate in the flip test does **not** mean the pairwise evaluator loses ~30 accuracy points. It means the raw scalar scorer has a positive additive offset, so many flipped-order scores remain above zero even though the relational component reverses.

The evaluator should therefore be described as:

```text
A strong fixed-order pairwise relational evaluator with scorer bias,
not a perfectly zero-centered antisymmetric comparator.
```

The relational preference encoding thesis remains intact.

---

## What the 95.2% result actually measured

The 95.2% result was measured in canonical HH-RLHF order:

```text
score(chosen, rejected) > 0
```

with the chosen response always supplied as the first argument and rejected as the second. Under that evaluation, the paper reports:

```text
Test accuracy:
  95.2% = 8,141 / 8,552

Average score:
  +1.6445

Score range:
  [-1.51, +4.56]
```

This is not the same as bidirectional antisymmetric accuracy.

It is fixed-order preference discrimination accuracy.

---

## Why 65% is not the corrected pairwise accuracy

The 65% figure belongs to a different experiment: the best nonlinear **independent / pointwise** evaluator.

That model sees one response representation at a time and tries to classify it as chosen or rejected without direct access to the paired response. The paper’s point is that this independent access is much weaker than pairwise relational access.

The important pattern is:

```text
Pairwise nonlinear evaluator:
  95.2%

Pairwise linear difference probe:
  84.5%

Independent nonlinear evaluator:
  ~65%

Independent linear classifier:
  21.75%, below chance, inverted polarity
```

So the 65% result is not a correction to the 95.2% result. It is part of the evidence that preference is much more accessible relationally than absolutely.

---

## What the evaluator actually computes

The pairwise evaluator is not an independent scorer of “chosen quality” and “rejected quality.” It is an ordered-pair scorer.

At each Ouro loop step, it roughly does:

```text
c_t = attention_pool(chosen_states_t)
r_t = attention_pool(rejected_states_t)
diff_t = c_t - r_t
normed_t = LayerNorm(diff_t, bias=False)
proj_t = Linear(normed_t)
```

Then it sends the sequence of projected differences through a GRU and produces one scalar score.

So the model is fundamentally relational:

```text
f(A, B)
```

not:

```text
score(A) - score(B)
```

This is why it is wrong to say “chosen gets +2.51 and rejected gets +2.51” as separate pointwise biases. The bias is on the ordered pairwise scalar output.

---

## What the flip test showed

The flip test computes:

```text
normal  = f(chosen, rejected)
flipped = f(rejected, chosen)
```

A perfectly antisymmetric raw scorer would satisfy:

```text
f(A, B) ≈ -f(B, A)
```

so:

```text
normal + flipped ≈ 0
```

But the reported epoch-2 model showed:

```text
antisymmetry correlation:
  ρ ≈ -0.94

strict sign flip rate:
  ~25%

mean(normal + flipped):
  +2.51
```

The key interpretation is:

```text
correlation is high:
  the relational component reverses under argument swap

strict sign flip is low:
  the scalar head has a positive offset, so flipped scores often stay positive
```

---

## The bias model

A simple model explains the result:

```text
f(A, B) = g(A, B) + b
f(B, A) = -g(A, B) + b
```

where:

```text
g(A, B):
  relational preference component

b:
  positive scalar scorer offset
```

Then:

```text
f(A, B) + f(B, A) = 2b
```

If the reported flip-test mean sum is:

```text
mean(normal + flipped) = +2.51
```

then the simplest shared-offset estimate is:

```text
b ≈ +1.255 per ordered call
```

If separate raw logs instead show that both normal and flipped orderings are each shifted by about +2.51, then the per-call offset is +2.51 and the sum should be about +5.02. Either way, the conclusion is the same: the offset applies to both ordered evaluations and causes strict sign flips to undercount relational order-sensitivity.

Example:

```text
g(chosen, rejected) = +1.0
b = +1.3

normal:
  f(chosen, rejected) = +2.3

flipped:
  f(rejected, chosen) = +0.3
```

The strict sign did not flip, but the relational component did.

---

## Why strict sign flip is not an accuracy metric

Strict sign flip asks:

```text
Does f(rejected, chosen) become negative?
```

But fixed-order HH-RLHF accuracy asks:

```text
Is f(chosen, rejected) positive?
```

Those are different questions.

Because the scalar head has a positive bias, strict sign flip is sensitive to calibration around zero. It is not a direct estimate of preference accuracy.

The paper’s cross-epoch result confirms this:

```text
Epoch 2:
  test accuracy = 95.2%
  strict sign flip = 25%
  mean sum bias = +2.51

Epoch 4/5:
  test accuracy collapses to ~67.2% / 62.4%
  strict sign flip rises to 96%
  scorer bias dissipates
```

So higher strict sign flip did not mean better preference learning. It mostly meant less positive scorer bias.

---

## Why the evaluator is not degenerate

The degenerate failure mode was a constant-output pairwise model:

```text
normal  ≈ +13
flipped ≈ +13
```

Such a model can pass fixed-order accuracy if chosen is always first, but it has no content dependence and fails the flip test.

The reported epoch-2 model is different:

```text
scores span:
  [-1.51, +4.56]

antisymmetry correlation:
  ρ ≈ -0.94

normal/flipped scores are content-dependent
```

So the model is not a constant positive-output trick. It has a strong relational signal plus scorer bias.

---

## Corrected interpretation

The correct statement is:

```text
The HH-RLHF evaluator reached 95.2% fixed-order pairwise preference accuracy.
The low strict flip rate does not reduce that to 65%.
The 65% figure belongs to independent pointwise evaluation.
The flip test exposes positive scorer bias in the raw pairwise scalar.
Antisymmetry correlation shows the relational component is strongly order-sensitive.
```

Or more compactly:

```text
The evaluator is real, but raw scores are biased.
The relational thesis survives.
For bidirectional comparator use, antisymmetrize or calibrate the score.
```

---

## Recommended audit

To fully settle the calibration concern, run an antisymmetrized HH-RLHF evaluator audit.

For each HH-RLHF test pair:

```text
normal  = f(chosen, rejected)
flipped = f(rejected, chosen)
antisym = (normal - flipped) / 2
bias    = (normal + flipped) / 2
```

Evaluate:

```text
raw_fixed_order_accuracy:
  normal > 0

antisymmetrized_accuracy:
  antisym > 0

bias_corrected_accuracy:
  normal - global_bias > 0

strict_sign_flip_rate:
  sign(normal) != sign(flipped)

antisymmetry_correlation:
  corr(normal, flipped)

margin-bin accuracy:
  accuracy by |antisym| bins

high-confidence subset:
  examples where |normal| or |antisym| is large
```

Expected outcomes:

```text
antisym accuracy near 93–95%:
  concern resolved; 95% is robustly relational

antisym accuracy around 85–92%:
  still strong, but fixed-order score benefited from bias/order

antisym accuracy much lower:
  serious issue; fixed-order result relied too much on canonical ordering
```

Given the reported `ρ ≈ -0.94`, the expectation is that antisymmetrized accuracy remains strong, but this should be measured directly.

---

## Practical implications for later branch/action selectors

For any pairwise tap or evaluator used as a general bidirectional comparator, do not blindly rely on raw one-way sign.

Prefer:

```text
s(A, B) = (f(A, B) - f(B, A)) / 2
```

or use pairwise tournament methods that explicitly account for scorer bias and calibration.

For fixed-order evaluation tasks, raw scores may remain valid if the ordering convention is consistent. For arbitrary branch comparison, terminal collapse, or action selection, calibration matters more.

---

## Final conclusion

The evaluator conversation resolves to this:

```text
The pairwise evaluator is not secretly 65%.
The 95.2% fixed-order HH-RLHF result is real.
The low flip-sign rate reflects scorer bias, not a 30-point accuracy collapse.
The 65% figure belongs to independent pointwise evaluation.
The relational preference encoding thesis remains supported.
The raw scorer should be antisymmetrized or calibrated before being used as a general bidirectional comparator.
```
