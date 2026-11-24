# Data layout

`kcat.csv` is the supplied 28,017-record dataset. Its contents are copied unchanged. `splits/split_random.csv` and `splits/split.csv` are the supplied random and homology assignments, copied unchanged.

The training target is `log10_value`. The original `fold` column is retained as source metadata and does not control this release's evaluation.

Raw ReactZyme sources, backbone model weights and generated arrays are not bundled. Consult the project README for the expected directories. Source paths can be overridden in a local configuration; generated caches remain project-local.
