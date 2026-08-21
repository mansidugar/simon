from tensorflow.keras.models import load_model

MODEL_PATH = "./simonNDvsDD/SENet/ND_VV_Simon32_9R.h5"

print("[+] Loading model...")

model = load_model(MODEL_PATH)

print("[+] Model loaded successfully")

print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)