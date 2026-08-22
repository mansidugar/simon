
````markdown
# Simon32/64 Explainability Experiments

This repository contains the Simon32/64 experiments conducted to evaluate the **cross-cipher generality of an explainability workflow developed for neural cryptanalysis**.

The primary study was conducted on Speck32/64. Simon32/64 was used as a secondary cipher to determine whether the same explainability stages could be applied beyond the primary cipher setting.

The Simon experiments focus on:

- Neural distinguisher evaluation
- Integrated Gradients attribution
- Single-feature causal ablation
- Attribution–causal faithfulness analysis
- Pairwise causal interaction analysis
- Round-to-round attribution transport analysis

---

## 1. Project Structure

The main Simon implementation is located in:

```text
speck32_simon32/
````

Important directories and files include:

```text
speck32_simon32/
│
├── simon/
│   ├── ND_VD_Simon32_8R.h5
│   ├── ND_VD_Simon32_9R.h5
│   ├── ND_VV_Simon32_8R.h5
│   ├── ND_VV_Simon32_9R.h5
│   ├── ND_VV_Simon32_10R.h5
│   ├── ND_VV_Simon32_11R.h5
│   └── ...
│
├── simonNDvsDD/
│   ├── DenseNet/
│   │   ├── 7_rounds_Simon_10depth.h5
│   │   ├── 8_rounds_Simon_10depth.h5
│   │   └── 9_rounds_Simon_10depth.h5
│   │
│   ├── ResNet_lr_1e-3_1e-5/
│   ├── ResNet_lr_2e-3_1e-4/
│   ├── SENet/
│   ├── simon.py
│   └── ...
│
├── 01_simon_9r_baseline.py
├── 02_simon_9r_ig_5000.py
├── 03_simon_9r_causal_5000.py
├── 04_simon_9r_faithfulness_5000.py
├── 05_simon_9r_pairwise_interaction_5000.py
├── 06_round_transport_simon.py
│
├── single_bit_causal_ablation_simon.py
├── phase2_ig_simon_densenet.py
├── multi_bit_interaction_simon.py
├── ig_vs_causal_correlation_simon.py
├── round_transport_simon.py
│
├── simon_9r_X_5000.npy
├── simon_9r_Y_5000.npy
├── simon_9r_ig_5000.npy
├── simon_9r_causal_5000.npy
├── simon_pairwise_synergy_5000.npy
│
└── README.md
```

---

# 2. Experimental Objective

The purpose of the Simon32/64 experiment was **not** to reproduce the complete Speck analysis.

Instead, it was used as an exploratory cross-cipher evaluation to determine whether the proposed explainability workflow could be transferred to another lightweight block cipher.

The Simon experiment therefore asks:

> Can attribution, causal validation, faithfulness analysis, and interaction analysis be applied to a neural distinguisher for a different cipher family?

The results are interpreted as evidence for **methodological generality**, rather than evidence that the same attribution patterns or the same degree of faithfulness must occur across different ciphers.

---

# 3. Model Used

The primary Simon experiment used the:

```text
9_rounds_Simon_10depth.h5
```

DenseNet model located at:

```text
simonNDvsDD/DenseNet/9_rounds_Simon_10depth.h5
```

The model operates on a:

```text
64-feature binary representation
```

of Simon32/64 ciphertext data.

---

# 4. Evaluation Dataset

The main Simon experiment used:

```text
Samples  : 5000
Features : 64
```

The evaluation dataset contained:

```text
Class 0 : 2519 samples
Class 1 : 2481 samples
```

The baseline 9-round Simon model obtained:

```text
Accuracy         : 63.94%
Mean confidence  : 0.142854
```

The exact evaluation dataset can be saved as:

```text
simon_9r_X_5000.npy
simon_9r_Y_5000.npy
```

These files should be retained when reproducibility of later experiments is required.

---

# 5. Integrated Gradients

Integrated Gradients was applied to the 9-round Simon DenseNet using:

```text
Baseline       : all-zero 64-dimensional vector
Samples        : 5000
Integration steps : 20
```

The resulting global attribution distribution was non-uniform.

The highest-attributed features were:

```text
Feature 20
Feature 19
Feature 21
Feature 22
Feature 18
```

with the corresponding attribution magnitudes:

```text
Feature 20 : 0.32900249
Feature 19 : 0.31693727
Feature 21 : 0.29029548
Feature 22 : 0.27435423
Feature 18 : 0.25962152
```

The complete attribution vector is saved as:

```text
simon_9r_ig_5000.npy
```

---

# 6. Single-Feature Causal Ablation

Each of the 64 input features was individually set to zero and the resulting classification accuracy was measured.

The baseline was:

```text
63.94%
```

The largest causal accuracy reduction was:

```text
Feature 22 : 7.98 percentage points
```

The next largest causal effects were:

```text
Feature 54 : 7.52 percentage points
Feature 60 : 7.38 percentage points
Feature 36 : 6.64 percentage points
Feature 4  : 6.40 percentage points
```

The complete causal-importance vector is saved as:

```text
simon_9r_causal_5000.npy
```

---

# 7. Attribution–Causal Faithfulness

The Integrated Gradients attribution ranking was compared with the causal-ablation ranking.

The observed results were:

```text
Spearman correlation : 0.262535
Spearman p-value     : 0.03609989

Pearson correlation  : 0.200347
Pearson p-value      : 0.1124380

Top-5 overlap        : 1/5
Top-10 overlap       : 2/10
Top-20 overlap       : 6/20
```

These results indicate a **relatively weak attribution–causal correspondence** for the Simon experiment.

The statistically significant Spearman association should not be interpreted as strong faithfulness because:

1. The correlation magnitude is modest.
2. The Pearson correlation is not statistically significant at the 0.05 level.
3. Top-k overlap is limited.

Therefore, the Simon experiment supports transferability of the attribution and causal workflow, but does **not** demonstrate that Integrated Gradients has equally strong causal agreement for Simon as observed in the primary Speck analysis.

---

# 8. Pairwise Interaction Analysis

Pairwise causal interactions were evaluated using the eight highest-ranked Integrated Gradients features:

```text
20
19
21
22
18
25
51
52
```

This produced:

```text
28 unique feature pairs
```

The interaction density was:

```text
Negative interactions : 25
Positive interactions : 3
Zero interactions     : 0
```

The strongest absolute interaction was:

```text
Features : (22,18)
Synergy  : -0.037200
Pair drop: 0.100200
```

Other strong negative interactions included:

```text
(21,22) : -0.028000
(18,25) : -0.024200
(21,18) : -0.021000
(22,25) : -0.018200
```

The strongest positive interaction reported in the experiment was:

```text
(20,52) : +0.005400
```

The saved interaction results are:

```text
simon_pairwise_synergy_5000.npy
```

Negative synergy indicates a less-than-additive joint effect under the adopted ablation-based interaction definition.

---

# 9. Round-to-Round Attribution Transport

An exploratory round-transport analysis was also conducted using Simon:

```text
7-round DenseNet
8-round DenseNet
9-round DenseNet
```

The analysis computed Integrated Gradients importance maps and compared them using Spearman correlation.

The saved importance maps are:

```text
simon_7R_importance.npy
simon_8R_importance.npy
simon_9R_importance.npy
```

The correlations are saved in:

```text
simon_round_transport_correlations.csv
```

The round-transport experiment is exploratory and should not be interpreted as a complete replication of the structural-stability analysis performed for DeepSpeck.

---

# 10. Reproducing the Main 9-Round Experiment

Run the following commands from the project root:

```bash
cd /Users/md/Documents/simon/speck32_simon32
```

Then execute the experiments in this order:

```bash
python3 01_simon_9r_baseline.py
```

```bash
python3 02_simon_9r_ig_5000.py
```

```bash
python3 03_simon_9r_causal_5000.py
```

```bash
python3 04_simon_9r_faithfulness_5000.py
```

```bash
python3 05_simon_9r_pairwise_interaction_5000.py
```

For the exploratory round-transport analysis:

```bash
python3 06_round_transport_simon.py
```

---

# 11. Generated Results

The main numerical outputs are:

```text
simon_9r_X_5000.npy
simon_9r_Y_5000.npy
simon_9r_ig_5000.npy
simon_9r_causal_5000.npy
simon_pairwise_synergy_5000.npy
```

These contain:

* fixed evaluation inputs
* evaluation labels
* Integrated Gradients attribution scores
* causal feature importance
* pairwise interaction results

---

# 12. Feature Indexing Convention

The Simon implementation uses **zero-based feature indexing**.

Therefore:

```text
Feature 0
Feature 1
...
Feature 63
```

are the actual indices reported by the implementation.

This convention is retained in the experimental outputs and manuscript discussion.

---

# 13. Scope and Interpretation

The Simon32/64 experiment is an **exploratory cross-cipher generality study**.

The results support the following conclusion:

> The proposed explainability workflow can be executed on a neural distinguisher for Simon32/64, producing attribution, causal, faithfulness, and interaction analyses.

The results do **not** establish that:

* the same features are important for Speck and Simon;
* the same attribution patterns transfer across ciphers;
* Integrated Gradients has the same causal agreement for all cipher families;
* the complete DeepSpeck framework has been replicated for Simon.

The Simon experiment therefore provides evidence for **methodological transferability**, while also demonstrating that the quantitative behavior of attribution and causal importance can depend on the underlying cipher and neural model.

---

# 14. Supporting Repository

The complete Simon implementation and supporting files are available at:

```text
https://github.com/mansidugar/simon/tree/main/speck32_simon32
```

This repository contains the Simon implementation, trained models, experiment scripts, numerical outputs, and supporting cryptanalysis files used by the exploratory evaluation.

```

### Download the README

[Download `README.md`](sandbox:/mnt/data/simon_saved_scripts/README_SIMON_EXPERIMENTS.md)

You can replace the README currently inside your `speck32_simon32` folder with this version and then commit it to your Simon repository.
```
