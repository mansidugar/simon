import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.stats import spearmanr
import sys
sys.path.append("./simonNDvsDD")
import simon

MODELS = [
    ("7R", "./simonNDvsDD/DenseNet/7_rounds_Simon_10depth.h5", 7),
    ("8R", "./simonNDvsDD/DenseNet/8_rounds_Simon_10depth.h5", 8),
    ("9R", "./simonNDvsDD/DenseNet/9_rounds_Simon_10depth.h5", 9),
]

N_SAMPLES = 100
IG_STEPS = 20

def integrated_gradients(model, sample, baseline, steps=20):
    alphas = np.linspace(0, 1, steps)
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

all_importance = {}

print()
print("========================================")
print("ROUND-TO-ROUND MOTIF TRANSPORT (SIMON)")
print("========================================")

for name, model_path, rounds in MODELS:
    print()
    print(f"[+] ANALYZING {name}")

    model = load_model(model_path, compile=False)
    X, _ = simon.make_train_data(1000, rounds)

    baseline = np.zeros((1, X.shape[1]), dtype=np.float32)
    importance = np.zeros(X.shape[1])

    for i in range(N_SAMPLES):
        sample = X[i:i+1].astype(np.float32)

        ig = integrated_gradients(
            model,
            sample,
            baseline,
            steps=IG_STEPS
        )

        importance += np.abs(ig.flatten())

    importance /= N_SAMPLES
    importance /= np.max(importance) + 1e-10
    all_importance[name] = importance

    print(f"[+] Top bit: {np.argmax(importance)}")

print()
print("========================================")
print("TRANSPORT CORRELATIONS")
print("========================================")

corr_78, p_78 = spearmanr(all_importance["7R"], all_importance["8R"])
corr_89, p_89 = spearmanr(all_importance["8R"], all_importance["9R"])
corr_79, p_79 = spearmanr(all_importance["7R"], all_importance["9R"])

print(f"7 -> 8 rounds : {corr_78:.6f}")
print(f"8 -> 9 rounds : {corr_89:.6f}")
print(f"7 -> 9 rounds : {corr_79:.6f}")

np.save("simon_7R_importance.npy", all_importance["7R"])
np.save("simon_8R_importance.npy", all_importance["8R"])
np.save("simon_9R_importance.npy", all_importance["9R"])

np.savetxt(
    "simon_round_transport_correlations.csv",
    np.array([
        [7, 8, corr_78, p_78],
        [8, 9, corr_89, p_89],
        [7, 9, corr_79, p_79],
    ]),
    delimiter=",",
    header="round_a,round_b,spearman_r,p_value",
    comments=""
)

print()
print("[+] Saved importance maps.")
print("[+] Saved: simon_round_transport_correlations.csv")
