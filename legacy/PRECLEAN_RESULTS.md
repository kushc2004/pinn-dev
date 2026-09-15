# Pre-clean protocol results

The original rebuild results remain available in git history, especially commit
`35908b568f0f2d98946d8b8de5263a8d4d85ca46`.

They are intentionally not kept as current result artifacts because the audit
performed on 2026-09-15 found two paths by which final-test information affected
model construction:

1. PySR sampled from the complete dataset, so some eventual high-load test rows
   and targets entered the symbolic surrogate.
2. The reported physics weight was chosen after comparing candidate lambdas on
   the same high-load test subset.

The current pipeline fixes both issues with train-only symbolic discovery,
validation-only lambda selection, protocol-specific scalers/equations, and a
one-shot final test stage.
