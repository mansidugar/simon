import os
import numpy as np
from tensorflow.keras.models import load_model
import sys
sys.path.append("./simonNDvsDD")
import simon

MODEL_PATH = "./simonNDvsDD/DenseNet/9_rounds_Simon_10depth.h5"
X_FILE = "simon_9r_X_5000.npy"
Y_FILE = "simon_9r_Y_5000.npy"
IG_FILE = "simon_9r_ig_5000.npy"

print("[+] Loading Simon 9-round DenseNet.")
model = load_model(MODEL_PATH, compile=False)

if os.path.exists(X_FILE) and os.path.exists(Y_FILE):
    X = np.load(X_FILE)
    Y = np.load(Y_FILE)
else:
    print("[+] Saved dataset not found. Generating a new one.")
    X, Y = simon.make_train_data(5000, 9)
    np.save(X_FILE, X)
    np.save(Y_FILE, Y)

ig = np.load(IG_FILE)

pred = model.predict(X, verbose=0).flatten()
baseline_acc = np.mean((pred > 0.5) == Y)
top_bits = np.argsort(ig)[::-1][:8]

print()
print("======================================")
print("SIMON 9R PAIRWISE INTERACTION")
print("======================================")
print(f"Samples          : {len(X)}")
print(f"Features         : {X.shape[1]}")
print(f"Baseline accuracy: {baseline_acc:.6f}")
print()
print("[+] Top 8 IG features:")
print(top_bits)

single_drop = {}
print()
print("[+] Computing single-feature drops.")

for bit in top_bits:
    X_mask = X.copy()
    X_mask[:, bit] = 0
    pred_mask = model.predict(X_mask, verbose=0).flatten()
    acc = np.mean((pred_mask > 0.5) == Y)
    single_drop[int(bit)] = baseline_acc - acc

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

        pred_pair = model.predict(X_pair, verbose=0).flatten()
        pair_acc = np.mean((pred_pair > 0.5) == Y)
        pair_drop = baseline_acc - pair_acc

        synergy = pair_drop - single_drop[b1] - single_drop[b2]
        results.append((b1, b2, synergy, pair_drop))

results.sort(key=lambda x: abs(x[2]), reverse=True)

print()
print("======================================")
print("TOP 15 CAUSAL INTERACTIONS")
print("======================================")
for rank, item in enumerate(results[:15], 1):
    b1, b2, syn, pd = item
    print(f"Rank {rank}: Features=({b1},{b2}) | Synergy={syn:.6f} | PairDrop={pd:.6f}")

values = np.array([r[2] for r in results])
positive = np.sum(values > 0)
negative = np.sum(values < 0)
zero = np.sum(values == 0)

print()
print("======================================")
print("INTERACTION DENSITY")
print("======================================")
print(f"Total tested pairs     : {len(results)}")
print(f"Positive synergy pairs : {positive}")
print(f"Negative synergy pairs : {negative}")
print(f"Zero synergy pairs     : {zero}")

np.save("simon_pairwise_synergy_5000.npy", np.array(results, dtype=object))
print()
print("[+] Saved: simon_pairwise_synergy_5000.npy")
