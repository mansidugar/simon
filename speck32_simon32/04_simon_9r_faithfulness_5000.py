import numpy as np
from scipy.stats import pearsonr, spearmanr

IG_FILE = "simon_9r_ig_5000.npy"
CAUSAL_FILE = "simon_9r_causal_5000.npy"

print("[+] Loading saved attribution and causal results.")
ig = np.load(IG_FILE).astype(np.float64)
causal = np.load(CAUSAL_FILE).astype(np.float64)

if ig.shape != causal.shape:
    raise ValueError(f"Shape mismatch: IG={ig.shape}, causal={causal.shape}")

ig_norm = ig / (np.max(ig) + 1e-10)
causal_norm = causal / (np.max(causal) + 1e-10)

def overlap(k):
    top_ig = set(np.argsort(ig_norm)[::-1][:k])
    top_causal = set(np.argsort(causal_norm)[::-1][:k])
    return len(top_ig.intersection(top_causal))

pearson_r, pearson_p = pearsonr(ig_norm, causal_norm)
spearman_r, spearman_p = spearmanr(ig_norm, causal_norm)

print()
print("======================================")
print("SIMON 9R FAITHFULNESS — 5000 SAMPLES")
print("======================================")
print(f"Features             : {len(ig_norm)}")
print(f"Pearson correlation  : {pearson_r:.6f}")
print(f"Pearson p-value      : {pearson_p:.6e}")
print(f"Spearman correlation: {spearman_r:.6f}")
print(f"Spearman p-value     : {spearman_p:.6e}")
print()
print(f"Top-5 overlap        : {overlap(5)}/5")
print(f"Top-10 overlap       : {overlap(10)}/10")
print(f"Top-20 overlap       : {overlap(20)}/20")

print()
print("TOP 10 IG FEATURES")
for rank, feature in enumerate(np.argsort(ig_norm)[::-1][:10], 1):
    print(f"Rank {rank}: Feature={feature} | IG={ig_norm[feature]:.8f}")

print()
print("TOP 10 CAUSAL FEATURES")
for rank, feature in enumerate(np.argsort(causal_norm)[::-1][:10], 1):
    print(f"Rank {rank}: Feature={feature} | Drop={causal_norm[feature]:.8f}")

print()
print("[+] Faithfulness analysis complete.")
