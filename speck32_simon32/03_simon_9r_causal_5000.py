import os
import numpy as np
from tensorflow.keras.models import load_model
import sys
sys.path.append("./simonNDvsDD")
import simon

MODEL_PATH = "./simonNDvsDD/DenseNet/9_rounds_Simon_10depth.h5"
X_FILE = "simon_9r_X_5000.npy"
Y_FILE = "simon_9r_Y_5000.npy"
N_SAMPLES = 5000
ROUNDS = 9

print("[+] Loading Simon 9-round DenseNet.")
model = load_model(MODEL_PATH, compile=False)

if os.path.exists(X_FILE) and os.path.exists(Y_FILE):
    X = np.load(X_FILE)
    Y = np.load(Y_FILE)
else:
    print("[+] Saved dataset not found. Generating a new one.")
    X, Y = simon.make_train_data(N_SAMPLES, ROUNDS)
    np.save(X_FILE, X)
    np.save(Y_FILE, Y)

print("[+] Computing baseline accuracy.")
pred = model.predict(X, verbose=0).flatten()
baseline_acc = np.mean((pred > 0.5) == Y)
print(f"[+] Baseline Accuracy: {baseline_acc:.6f}")
print(f"[+] Baseline Accuracy (%): {baseline_acc * 100:.2f}%")

causal_importance = np.zeros(X.shape[1], dtype=np.float64)
print()
print("[+] Performing single-feature ablations.")

for bit in range(X.shape[1]):
    X_masked = X.copy()
    X_masked[:, bit] = 0
    pred_masked = model.predict(X_masked, verbose=0).flatten()
    masked_acc = np.mean((pred_masked > 0.5) == Y)
    drop = baseline_acc - masked_acc
    causal_importance[bit] = drop
    print(f"Feature {bit:02d} | Accuracy={masked_acc:.6f} | Drop={drop:.6f}")

np.save("simon_9r_causal_5000.npy", causal_importance)

top_bits = np.argsort(causal_importance)[::-1][:20]

print()
print("======================================")
print("SIMON 9R CAUSAL ABLATION")
print("======================================")
print(f"Features            : {X.shape[1]}")
print(f"Baseline accuracy   : {baseline_acc:.6f}")
print()
print("TOP 20 CAUSAL FEATURES")
for rank, bit in enumerate(top_bits, 1):
    print(f"Rank {rank}: Feature={bit} | Drop={causal_importance[bit]:.6f}")
print()
print(f"Maximum causal drop: {np.max(causal_importance):.6f}")
print("[+] Saved: simon_9r_causal_5000.npy")
