from tensorflow.keras.models import load_model

models = [
"./simonNDvsDD/SENet/ND_VV_Simon32_7R.h5",
"./simonNDvsDD/SENet/ND_VV_Simon32_8R.h5",
"./simonNDvsDD/SENet/ND_VV_Simon32_9R.h5",
"./simonNDvsDD/SENet/ND_VV_Simon32_10R.h5",
"./simonNDvsDD/SENet/ND_VV_Simon32_11R.h5",
"./simonNDvsDD/DenseNet/7_rounds_Simon_10depth.h5",
"./simonNDvsDD/DenseNet/8_rounds_Simon_10depth.h5",
"./simonNDvsDD/DenseNet/9_rounds_Simon_10depth.h5",
]

for model_path in models:

    print("\nTrying:", model_path)

    try:
        model = load_model(
            model_path,
            compile=False
        )

        print("SUCCESS")

    except Exception as e:

        print("FAILED")
        print(type(e).__name__)
        print(e)