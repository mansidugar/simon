import numpy as np
from tensorflow.keras.models import load_model
import sys
sys.path.append("./simonNDvsDD")
import simon

MODEL_PATH = "./simonNDvsDD/DenseNet/9_rounds_Simon_10depth.h5"
N_SAMPLES = 5000
ROUNDS = 9

print("[+] Loading Simon 9-round DenseNet.")
model = load_model(MODEL_PATH, compile=False)

print("[+] Generating evaluation dataset.")
X, Y = simon.make_train_data(N_SAMPLES, ROUNDS)

print("[+] Computing baseline predictions.")
pred = model.predict(X, verbose=0).flatten()
pred_labels = (pred > 0.5).astype(np.uint8)
accuracy = np.mean(pred_labels == Y)
mean_confidence = np.mean(np.abs(pred - 0.5))

np.save("simon_9r_X_5000.npy", X)
np.save("simon_9r_Y_5000.npy", Y)

print()
print("======================================")
print("SIMON 9R BASELINE")
print("======================================")
print(f"Samples            : {len(X)}")
print(f"Features           : {X.shape[1]}")
print(f"Class 0            : {np.sum(Y == 0)}")
print(f"Class 1            : {np.sum(Y == 1)}")
print(f"Baseline Accuracy  : {accuracy:.6f}")
print(f"Baseline Accuracy% : {accuracy * 100:.2f}%")
print(f"Mean Confidence    : {mean_confidence:.6f}")
print()
print("[+] Saved:")
print("    simon_9r_X_5000.npy")
print("    simon_9r_Y_5000.npy")
