# Conference integration sketch

Add a short “Visible-prefix conditionality gate” subsection to the conference
extension, not to the frozen arXiv package.

Suggested text:

> We preregistered a zero-generation comparison between a strong external
> prompt-plus-prefix reader and the frozen hidden-state readout. The endpoint
> could not be adjudicated from the preserved cohorts. GSM8K did not preserve
> exact candidate prefix text or the required original three-way split. In
> Horizon Logic, deterministic reconstruction showed that almost 90% of
> scorable tasks had at least one pre-marker prefix that explicitly stated the
> normalized gold conclusion. Taskwise exclusion exceeded the preregistered
> materiality threshold and left an under-sized held-out arm. We therefore did
> not fit or compare V, H, and HV, did not test equivalence, and retain the
> published hidden-versus-shortcut results only as historical sensitivities.

Use the exact verdict labels:

- Horizon new-only: `VISIBLE_PREFIX_NOT_ADJUDICABLE`
- GSM8K: `VISIBLE_PREFIX_NOT_ADJUDICABLE`
- Horizon pooled: `VISIBLE_PREFIX_NOT_ADJUDICABLE`
- aggregate: `VISIBLE_PREFIX_AGGREGATE_NOT_ADJUDICABLE`

Do not write that the visible reader explains the hidden effect, that the
hidden readout has a conditional increment, that the constructions are
equivalent, or that visible text wins. The next valid experiment needs
prospective preservation of token IDs, a task-disjoint three-way split, and a
cut placed before any semantic commitment to the answer—not merely before the
literal `FINAL ANSWER` marker.
