import numpy as np
from scipy.stats import spearmanr

print("[+] Loading results.")

ig = np.load(
    "simon_densenet_9r_ig.npy"
)

causal = np.load(
    "simon_single_bit_causal.npy"
)

# ----------------------------------
# Normalize
# ----------------------------------

ig = ig / (
    np.max(ig) + 1e-10
)

causal = causal / (
    np.max(causal) + 1e-10
)

# ----------------------------------
# Correlation
# ----------------------------------

corr, pval = spearmanr(
    ig,
    causal
)

# ----------------------------------
# Top-10 overlap
# ----------------------------------

top_ig = np.argsort(
    ig
)[::-1][:10]

top_causal = np.argsort(
    causal
)[::-1][:10]

overlap = len(
    set(top_ig).intersection(
        set(top_causal)
    )
)

# ----------------------------------
# Top-20 overlap
# ----------------------------------

top_ig20 = np.argsort(
    ig
)[::-1][:20]

top_causal20 = np.argsort(
    causal
)[::-1][:20]

overlap20 = len(
    set(top_ig20).intersection(
        set(top_causal20)
    )
)

print()
print("======================================")
print("SIMON FAITHFULNESS ANALYSIS")
print("======================================")

print(
    f"IG vs Causal Correlation : {corr:.6f}"
)

print(
    f"P-value                  : {pval:.6f}"
)

print()

print(
    f"Top-10 overlap : {overlap}/10"
)

print(
    f"Top-20 overlap : {overlap20}/20"
)

print()

print("TOP IG BITS")

for bit in top_ig:

    print(
        f"Bit={bit} "
        f"IG={ig[bit]:.6f}"
    )

print()

print("TOP CAUSAL BITS")

for bit in top_causal:

    print(
        f"Bit={bit} "
        f"Drop={causal[bit]:.6f}"
    )

# ----------------------------------
# Strong agreement bits
# ----------------------------------

agreement = ig * causal

best_agreement = np.argsort(
    agreement
)[::-1][:10]

print()
print("TOP AGREEMENT BITS")

for bit in best_agreement:

    print(
        f"Bit={bit} "
        f"Agreement={agreement[bit]:.6f}"
    )

print()
print("[+] Faithfulness analysis complete.")