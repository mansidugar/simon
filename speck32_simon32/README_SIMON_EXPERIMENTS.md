# Simon32/64 Explainability Experiments

These scripts save the terminal-based Simon experiments as reusable files.

## Run location

Run from the Simon project root:

```bash
cd /Users/md/Documents/simon/speck32_simon32
```

The scripts expect:

```text
simonNDvsDD/simon.py
simonNDvsDD/DenseNet/9_rounds_Simon_10depth.h5
simonNDvsDD/DenseNet/8_rounds_Simon_10depth.h5
simonNDvsDD/DenseNet/7_rounds_Simon_10depth.h5
```

## Main 9R workflow

Run in this order:

```bash
python3 01_simon_9r_baseline.py
python3 02_simon_9r_ig_5000.py
python3 03_simon_9r_causal_5000.py
python3 04_simon_9r_faithfulness_5000.py
python3 05_simon_9r_pairwise_interaction_5000.py
```

The baseline script saves the exact 5,000-sample evaluation set:

```text
simon_9r_X_5000.npy
simon_9r_Y_5000.npy
```

Later 9R scripts reuse these files when available.

Main outputs:

```text
simon_9r_ig_5000.npy
simon_9r_causal_5000.npy
simon_pairwise_synergy_5000.npy
```

## Round transport

Run:

```bash
python3 06_round_transport_simon.py
```

Outputs:

```text
simon_7R_importance.npy
simon_8R_importance.npy
simon_9R_importance.npy
simon_round_transport_correlations.csv
```

## Indexing convention

The Simon implementation reports feature positions using **zero-based indexing**. Preserve that convention when interpreting or citing the saved outputs.

## Reproducibility

The Simon data generator uses random sampling. Keep the saved 9R evaluation arrays with the repository if you need later analyses to use the exact same samples.
