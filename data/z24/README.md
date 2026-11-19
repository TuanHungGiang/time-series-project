# Z24

Source: [KU Leuven Z24 Bridge benchmark](https://bwk.kuleuven.be/bwm/z24). Obtain access under the source terms. Put the prepared `kaggle_z24_split` root under `processed/`, including `metadata/records_manifest.csv` and the corresponding split folders with MAT files. Point `--z24-root` to that root.

The runner fits on 24 healthy training records across nine setups and AVT/FVT. Within each record it reserves trailing windows as a healthy reference. Healthy validation has six records. The test cohort has six healthy and 126 damaged records. Record scores are medians of window scores. Sensor identities align within each setup and excitation group. The existing manifest defines that cohort.
