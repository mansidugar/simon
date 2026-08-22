import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
sys.path.append("./simonNDvsDD")
import simon

MODEL_PATH = "./simonNDvsDD/DenseNet/9_rounds_Simon_10depth.h5"
X_FILE = "simon_9r_X_5000.npy"
N_SAMPLES = 5000
STEPS = 20

def integrated_gradients(model, sample, baseline, steps=20):
    alphas = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    grads = []
    for alpha in alphas:
        x = baseline + alpha * (sample - baseline)
        x = tf.cast(x, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            pred = tf.reduce_sum(model(x, training=False))
        grad = tape.gradient(pred, x)
        grads.append(grad.numpy())
    avg_grad = np.mean(grads, axis=0)
    return (sample - baseline) * avg_grad

print("[+] Loading Simon 9-round DenseNet.")
model = load_model(MODEL_PATH, compile=False)

if os.path.exists(X_FILE):
    X = np.load(X_FILE)
else:
    print("[+] Saved dataset not found. Generating one.")
    X, _ = simon.make_train_data(N_SAMPLES, 9)

print("[+] Computing Integrated Gradients.")
print(f"[+] Samples: {len(X)}")
print(f"[+] Steps: {STEPS}")
print("[+] Baseline: all-zero vector")

baseline = np.zeros((1, X.shape[1]), dtype=np.float32)
importance = np.zeros(X.shape[1], dtype=np.float64)

for i in range(len(X)):
    sample = X[i:i+1].astype(np.float32)
    ig = integrated_gradients(model, sample, baseline, steps=STEPS)
    importance += np.abs(ig.flatten())
    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{len(X)}")

importance /= len(X)
np.save("simon_9r_ig_5000.npy", importance)

top = np.argsort(importance)[::-1][:20]

print()
print("======================================")
print("SIMON 9R INTEGRATED GRADIENTS")
print("======================================")
print(f"Input features : {len(importance)}")
print(f"IG steps       : {STEPS}")
print(f"Samples        : {len(X)}")
print(f"Max importance : {np.max(importance):.8f}")
print(f"Mean importance: {np.mean(importance):.8f}")
print()
print("TOP 20 IG FEATURES")
for rank, feature in enumerate(top, 1):
    print(f"Rank {rank}: Feature={feature} | IG={importance[feature]:.8f}")
print()
print("[+] Saved: simon_9r_ig_5000.npy")
