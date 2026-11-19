# HellBridge

Source: [NTNU Hell Bridge Test Arena dataset](https://doi.org/10.5281/zenodo.14632942). Place `data_100Hz.h5` under `raw/` or pass its path with `--hell-h5`.

The runner selects `P2_NM_Z` and the Svendsen-style window allocation. It uses 40 vertical acceleration channels, 60 healthy fit windows, 20 healthy calibration windows, 80 healthy test windows, and 640 damage test windows. The healthy windows come from two recordings shared across development and test. Evaluation is window-level.
