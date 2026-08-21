import numpy as np
from tensorflow.keras.models import load_model

import sys
sys.path.append("./simonNDvsDD")

import simon

MODEL_PATH = "./simonNDvsDD/DenseNet/9_rounds_Simon_10depth.h5"

print("[+] Loading model.")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("[+] Model loaded.")

# -------------------------
# Load previous results
# -------------------------

ig = np.load(
    "simon_densenet_9r_ig.npy"
)

# -------------------------
# Generate dataset
# -------------------------

print("[+] Generating dataset.")

X, Y = simon.make_train_data(
    1000,
    9
)

# -------------------------
# Baseline Accuracy
# -------------------------

pred = model.predict(
    X,
    verbose=0
).flatten()

baseline_acc = np.mean(
    (pred > 0.5) == Y
)

print(
    f"[+] Baseline Accuracy: {baseline_acc:.6f}"
)

# -------------------------
# Top IG bits
# -------------------------

top_bits = np.argsort(
    ig
)[::-1][:8]

print()
print("[+] Top IG bits:")
print(top_bits)

# -------------------------
# Single-bit drops
# -------------------------

single_drop = {}

print()
print("[+] Computing single-bit drops.")

for bit in top_bits:

    X_mask = X.copy()

    X_mask[:, bit] = 0

    pred_mask = model.predict(
        X_mask,
        verbose=0
    ).flatten()

    acc = np.mean(
        (pred_mask > 0.5) == Y
    )

    drop = baseline_acc - acc

    single_drop[bit] = drop

# -------------------------
# Pairwise interactions
# -------------------------

results = []

print()
print("[+] Computing pairwise interactions.")

for i in range(len(top_bits)):

    for j in range(i + 1, len(top_bits)):

        b1 = int(top_bits[i])
        b2 = int(top_bits[j])

        X_pair = X.copy()

        X_pair[:, b1] = 0
        X_pair[:, b2] = 0

        pred_pair = model.predict(
            X_pair,
            verbose=0
        ).flatten()

        pair_acc = np.mean(
            (pred_pair > 0.5) == Y
        )

        pair_drop = baseline_acc - pair_acc

        synergy = (
            pair_drop
            - single_drop[b1]
            - single_drop[b2]
        )

        results.append(
            (
                b1,
                b2,
                synergy,
                pair_drop
            )
        )

results.sort(
    key=lambda x: abs(x[2]),
    reverse=True
)

# -------------------------
# Output
# -------------------------

print()
print("======================================")
print("TOP CAUSAL INTERACTIONS")
print("======================================")

for rank, item in enumerate(
        results[:15],
        start=1):

    b1, b2, syn, pd = item

    print(
        f"Rank {rank}: "
        f"Bits=({b1},{b2}) | "
        f"Synergy={syn:.6f} | "
        f"PairDrop={pd:.6f}"
    )

# -------------------------
# Statistics
# -------------------------

positive = np.sum(
    np.array([r[2] for r in results]) > 0
)

negative = np.sum(
    np.array([r[2] for r in results]) < 0
)

print()
print("======================================")
print("INTERACTION DENSITY")
print("======================================")

print(
    f"Positive synergy pairs : {positive}"
)

print(
    f"Negative synergy pairs : {negative}"
)

np.save(
    "simon_pairwise_synergy.npy",
    np.array(results, dtype=object)
)

print()
print(
    "[+] Saved: simon_pairwise_synergy.npy"
)