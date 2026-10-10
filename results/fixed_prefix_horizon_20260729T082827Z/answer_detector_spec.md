# Answer-bearing detector specification: not instantiated

Status: `FIXED_PREFIX_HORIZON_NOT_ADJUDICABLE`

The detector was required to operate on the first N **preserved generated
tokens**. Because none of the 2,720 candidates preserves those IDs, no candidate
boundary can be rendered exactly without prohibited retokenization. Detector
implementation and training-only hand correction therefore did not begin.

The requested detection categories and grid remain design requirements, not
executed analyses. No answer spans were censored and no final-marker cut was
reused.
