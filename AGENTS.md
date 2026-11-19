# Repository guide

Read the root README, `.agent/state.md`, and the active experiment notes before editing. Keep reusable code in `src/clmm_timesnet/` and notebooks thin. Keep immutable dataset inputs under `data/<dataset>/` by processing stage. Record dataset provenance, exact inputs, seeds, masks, and protocol differences.

Keep outputs under a new experiment directory with a resolved config, a subdirectory per dataset, and a cross-dataset summary. Track small evidence files intentionally and keep raw data, checkpoints, and generated logs out of Git. Keep English and Vietnamese human-facing docs aligned. Keep any future Elsevier manuscript in one `paper/main.tex` file after its publication is authorized.

After a meaningful change, update `.agent/state.md`, add a durable decision when necessary, and document changes under Unreleased in `CHANGELOG.md`.
