# Repository structure

[Tiếng Việt](../vi/repository-structure.md)

Inputs are organized as `data/<dataset>/<stage>/`. Keep raw files immutable. Reusable code lives in `src/clmm_timesnet/`, runnable starting configurations in `configs/`, and the Kaggle entry point in `notebooks/analysis/`. Outputs are organized as `experiments/<run>/<dataset>/` and `experiments/<run>/summary/`. Create a new output directory for each execution.
