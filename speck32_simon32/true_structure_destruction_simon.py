import numpy as np
from tensorflow.keras.models import load_model
from scipy.stats import spearmanr

import sys
sys.path.append("./simonNDvsDD")

import simon

MODEL_PATH = "./simonNDvsDD/DenseNet/9_rounds_Simon_10depth.h5"

print()
print("========================================")
print("TRUE STRUCTURE DESTRUCTION CONTROL")
print("========================================")

print("[+] Loading model.")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("[+] Model loaded.")

# --------------------------------
# REAL DATA
# --------------------------------

print("[+] Generating REAL dataset.")

X_real, Y_real = simon.make_train_data(
    1000,
    9
)

pred = model.predict(
    X_real,
    verbose=0
).flatten()

correct_mask = (
    (pred > 0.5) == Y_real
)

X_real = X_real[correct_mask]

print(
    f"[+] Correctly classified samples: {len(X_real)}"
)

# --------------------------------
# REAL IMPORTANCE
# --------------------------------

print("[+] Computing REAL structure.")

real_importance = np.zeros(
    X_real.shape[1]
)

for bit in range(X_real.shape[1]):

    X_mask = X_real.copy()

    X_mask[:, bit] = 0

    pred_mask = model.predict(
        X_mask,
        verbose=0
    ).flatten()

    real_importance[bit] = np.mean(
        np.abs(
            pred[correct_mask]
            -
            pred_mask
        )
    )

real_importance /= (
    np.max(real_importance)
    + 1e-10
)

# --------------------------------
# RANDOM DATA
# --------------------------------

print("[+] Creating TRUE RANDOM dataset.")

X_rand = np.random.randint(
    0,
    2,
    size=X_real.shape
)

print("[+] Computing RANDOM structure.")

random_importance = np.zeros(
    X_rand.shape[1]
)

base_rand = model.predict(
    X_rand,
    verbose=0
).flatten()

for bit in range(X_rand.shape[1]):

    X_mask = X_rand.copy()

    X_mask[:, bit] = 0

    pred_mask = model.predict(
        X_mask,
        verbose=0
    ).flatten()

    random_importance[bit] = np.mean(
        np.abs(
            base_rand
            -
            pred_mask
        )
    )

random_importance /= (
    np.max(random_importance)
    + 1e-10
)

# --------------------------------
# COMPARISON
# --------------------------------

corr, pval = spearmanr(
    real_importance,
    random_importance
)

entropy_real = -np.sum(
    real_importance
    *
    np.log(
        real_importance + 1e-10
    )
)

entropy_rand = -np.sum(
    random_importance
    *
    np.log(
        random_importance + 1e-10
    )
)

print()
print("========================================")
print("TRUE STRUCTURE DESTRUCTION RESULTS")
print("========================================")

print(
    f"REAL entropy           : {entropy_real:.6f}"
)

print(
    f"RANDOM entropy         : {entropy_rand:.6f}"
)

print()
print(
    f"REAL vs RANDOM correlation : {corr:.6f}"
)

print(
    f"P-value                    : {pval:.6f}"
)

print()
print("========================================")
print("TOP REAL HOTSPOTS")
print("========================================")

for rank, bit in enumerate(
        np.argsort(real_importance)[::-1][:10],
        start=1):

    print(
        f"Rank {rank}: "
        f"Bit={bit}, "
        f"Importance={real_importance[bit]:.6f}"
    )

print()
print("========================================")
print("TOP RANDOM HOTSPOTS")
print("========================================")

for rank, bit in enumerate(
        np.argsort(random_importance)[::-1][:10],
        start=1):

    print(
        f"Rank {rank}: "
        f"Bit={bit}, "
        f"Importance={random_importance[bit]:.6f}"
    )

np.save(
    "simon_real_importance.npy",
    real_importance
)

np.save(
    "simon_random_importance.npy",
    random_importance
)

print()
print("[+] True structure destruction complete.")