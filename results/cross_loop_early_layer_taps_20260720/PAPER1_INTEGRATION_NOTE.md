# Paper 1 integration note (does NOT edit the paper)

This note proposes; it changes no paper file, style file, or figure. Another process owns
the paper edits.

## Does the result merit inclusion?

**Yes, as a compact readout-geometry result** — one short subsection or a boxed paragraph
plus one figure in the locus/readout-geometry material, adjacent to wherever the current
24/36/47 canonical basis and the "readable but frozen-intervention-negative" discussion
live. It is not a headline capability claim and should not be promoted to one.

## Narrowest supported claim

> On the CoreContent task-disjoint candidate-quality readout, a matched low-capacity
> antisymmetric tap recovers process-quality information at physical layers 8 and 16 during
> recurrent loops L3–L4 at parity with the layer-24 readout of the same loop and with the
> historical L4_36/L4_47 basis (heldout macro top-1 ≈ 0.62–0.63 vs chance 0.28; task-
> clustered CIs exclude chance). Readability increases monotonically across loops (L4−L1 ≈
> +0.19–0.21 at every layer), and the carrying direction is coordinate-stable across loops
> from L2 onward (frozen L2/L3→L4 transplant retains ≥97% of the target-local refit,
> Spearman 0.90–0.99), while the first loop's direction rotates (L1→later transplant loses
> 0.11–0.23). This is a readout-geometry result; no intervention was performed.

## Proposed location

The readout-geometry / locus section that currently introduces layers 24/36/47 as the
canonical per-loop basis and notes that frozen interventions did not yield a validated gain.
This result refines that story: it identifies *where and when* the signal is readable
earliest, and separates "readable" from "coordinate-stable."

## Proposed table / figure

- One figure: `fig1_local_refit_heatmap` (loop × layer readability), optionally with
  `fig4_transplant_gap` as a supplementary panel for the rotation story.
- One small table: the 3×4 local-refit macro top-1 grid with the L4_36/L4_47 reference row
  and the L2+/L1 transfer-stability contrast.

## Limitations to state alongside it

- Bounded subset (120 heldout groups/domain); one preregistered pooling convention.
- Alignment is 2-candidate pairwise (0.5 floor) and dominates the label-shuffle floor;
  reported per-domain and with the shuffle control.
- 195 hendrycks_math task-ID collisions were excluded; flags a dataset-hygiene fix for any
  future v3 build.
- Exploratory across 14 cells; the finding rests on the *consistent monotone loop trend and
  the L1-vs-L2+ stability split*, not on any single cell clearing significance.
- S3B2 corroboration is small-n (16 tasks) and the frozen CoreContent direction does **not**
  transfer to generated branches.

## Does it change the readout–control interpretation?

It **sharpens** it without overturning it. It does not create a control result. It does
identify a **promising intervention surface** — early-layer (8/16), late-loop (L3/L4) cells
that are readable while leaving 32–40 layers of same-loop downstream depth — which is exactly
the profile the motivating hypothesis wanted. But this experiment is **readout-only**: it
tests neither controllability nor a capability gain. Any claim that these loci are actionable
requires a separate, properly-controlled intervention experiment. The paper should describe
them as candidate readout loci, not steering loci.

## Verdict on what this identifies

Identifies **only a promising intervention locus (readout-only)**. It does **not** test
control, does not close the detection→selection gap, and must not be cited as
CONTROL_SOLVED, ACTIONABLE, or STEERING_LOCUS_FOUND.
