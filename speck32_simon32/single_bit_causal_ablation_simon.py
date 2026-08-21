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

print("[+] Model loaded successfully.")

# ----------------------------
# Generate evaluation dataset
# ----------------------------

print("[+] Generating dataset.")

X, Y = simon.make_train_data(
    1000,
    9
)

# ----------------------------
# Baseline Accuracy
# ----------------------------

pred = model.predict(
    X,
    verbose=0
).flatten()

baseline_acc = np.mean(
    (pred > 0.5) == Y
)

print()
print(
    f"[+] Baseline Accuracy: {baseline_acc:.6f}"
)

# ----------------------------
# Single-Bit Ablation
# ----------------------------

num_bits = X.shape[1]

causal_importance = np.zeros(
    num_bits
)

print()
print("[+] Performing single-bit ablations.")

for bit in range(num_bits):

    X_masked = X.copy()

    X_masked[:, bit] = 0

    pred_masked = model.predict(
        X_masked,
        verbose=0
    ).flatten()

    masked_acc = np.mean(
        (pred_masked > 0.5) == Y
    )

    drop = baseline_acc - masked_acc

    causal_importance[bit] = drop

    print(
        f"Bit {bit:02d} | "
        f"Accuracy={masked_acc:.6f} | "
        f"Drop={drop:.6f}"
    )

# ----------------------------
# Save Results
# ----------------------------

np.save(
    "simon_single_bit_causal.npy",
    causal_importance
)

# ----------------------------
# Summary
# ----------------------------

print()
print("======================================")
print("TOP CAUSALLY IMPORTANT BITS")
print("======================================")

top_bits = np.argsort(
    causal_importance
)[::-1][:10]

for rank, bit in enumerate(
        top_bits,
        start=1):

    print(
        f"Rank {rank}: "
        f"Bit={bit}, "
        f"CausalDrop={causal_importance[bit]:.6f}"
    )

print()
print(
    "[+] Saved: simon_single_bit_causal.npy"
)