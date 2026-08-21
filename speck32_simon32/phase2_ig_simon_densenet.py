import numpy as np
import tensorflow as tf
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

print("[+] Generating dataset.")

X, Y = simon.make_train_data(
    5000,
    9
)

print("[+] Dataset shape:", X.shape)

baseline = np.zeros(
    (1, X.shape[1]),
    dtype=np.float32
)

def integrated_gradients(
        model,
        sample,
        baseline,
        steps=20):

    alphas = np.linspace(
        0,
        1,
        steps
    )

    grads = []

    for alpha in alphas:

        x = baseline + alpha * (
            sample - baseline
        )

        x = tf.cast(
            x,
            tf.float32
        )

        with tf.GradientTape() as tape:

            tape.watch(x)

            pred = model(
                x,
                training=False
            )

            pred = tf.reduce_sum(pred)

        grad = tape.gradient(
            pred,
            x
        )

        grads.append(
            grad.numpy()
        )

    avg_grad = np.mean(
        grads,
        axis=0
    )

    ig = (
        sample - baseline
    ) * avg_grad

    return ig

print("[+] Computing IG.")

importance = np.zeros(
    X.shape[1]
)

N = 500

for i in range(N):

    sample = X[i:i+1].astype(
        np.float32
    )

    ig = integrated_gradients(
        model,
        sample,
        baseline
    )

    importance += np.abs(
        ig.flatten()
    )

    if i % 50 == 0:

        print(
            f"Processed {i}/{N}"
        )

importance /= N

np.save(
    "simon_densenet_9r_ig.npy",
    importance
)

print()
print("====================================")
print("SIMON IG ANALYSIS COMPLETE")
print("====================================")

print(
    "Input dimension:",
    len(importance)
)

print(
    "Max importance:",
    np.max(importance)
)

print(
    "Min importance:",
    np.min(importance)
)

print(
    "Mean importance:",
    np.mean(importance)
)

top_bits = np.argsort(
    importance
)[::-1][:10]

print()
print("TOP 10 IMPORTANT BITS")

for rank, bit in enumerate(
        top_bits,
        start=1):

    print(
        f"Rank {rank}: "
        f"Bit={bit}, "
        f"Score={importance[bit]:.6f}"
    )

print()
print(
    "[+] Saved: simon_densenet_9r_ig.npy"
)