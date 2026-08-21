import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import simon

MODEL = "./SENet/ND_VV_Simon32_9R.h5"

net = load_model(MODEL)

X, Y = simon.make_train_data(
    n=5000,
    nr=9
)

baseline = np.zeros_like(X[:1])

def integrated_gradients(
        model,
        sample,
        baseline,
        steps=50):

    alphas = np.linspace(0,1,steps)

    grads = []

    for alpha in alphas:

        x = baseline + alpha*(sample-baseline)

        x = tf.cast(x, tf.float32)

        with tf.GradientTape() as tape:

            tape.watch(x)

            pred = model(x)

        grad = tape.gradient(pred,x)

        grads.append(grad.numpy())

    avg_grad = np.mean(grads,axis=0)

    ig = (sample-baseline)*avg_grad

    return ig


importance = np.zeros(64)

for i in range(1000):

    sample = X[i:i+1]

    ig = integrated_gradients(
            net,
            sample,
            baseline)

    importance += np.abs(
        ig.reshape(-1)
    )

importance /= 1000

np.save(
    "simon_vv_9r_ig.npy",
    importance
)

print("Saved IG scores")