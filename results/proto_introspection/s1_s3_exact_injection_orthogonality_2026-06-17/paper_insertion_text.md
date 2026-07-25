# Paper Insertion Text

## Cautious Main Body

We regenerated the frozen S1.4 branch/carry protocol from the saved June-17 configuration (4 tasks, 12 loci, K=2, alpha=0.02, fixed last-token perturbations) and captured the actual boundary deltas used by the branch/carry mechanism. In the regenerated bundle, the verifier-labeled outcome direction projects 0.018296 of its squared norm into the actual injection/carry span, leaving residual 0.981704. Because the original historical tensors were not persisted, this is an exact-protocol regeneration rather than a historical tensor replay; nevertheless it directly audits the S1.4 writable frozen directions.

## Stronger Main Body

The exact S1.4 protocol regeneration closes the geometry gap: the frozen branch/carry deltas span a low-rank writable subspace, but the verifier-success direction lies almost entirely outside it (projection 0.018296, residual 0.981704). This supports the readout-control boundary interpretation: the model exposes process-quality information in hidden states, while the frozen write directions available to the branch/carry mechanism are not aligned with the outcome-relevant geometry.

## Compute Pitch

This geometry makes S3A a training-time alignment test rather than another frozen-control tweak. The frozen mechanism is mechanically valid and K-matched sampling closes the apparent fork gain, but the actual writable branch/carry span is misaligned with verifier-success directions. The next compute should therefore train on verifier-labeled generated branches to align readable process-state directions with controllable branch dynamics.
