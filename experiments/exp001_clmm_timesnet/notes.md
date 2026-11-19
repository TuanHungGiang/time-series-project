# CLMM–TimesNet on two bridges

Status: protocol registered. Comparative real-data runs are pending.

Run the root README command with this directory's `config.yaml` and a fresh output path. The snapshot fixes seed 20270905, 30 TimesNet epochs, 20 adapter epochs, and five nested masks per nonzero loss rate. See [the protocol](../../docs/en/experiment-protocol.md) for ablation definitions.

Review `summary/paired_deltas_summary.csv` for the incremental effect of attention and spectral loss, then compare the full variant with TimesNet on each bridge. Report recall and false positives jointly at the clean healthy threshold.
