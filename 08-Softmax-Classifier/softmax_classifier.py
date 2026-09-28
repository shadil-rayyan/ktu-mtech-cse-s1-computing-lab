import numpy as np

import tensorflow as tf

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# Convert labels into 1D array
y_train = y_train.flatten()
y_test = y_test.flatten()


print("Training samples:", x_train.shape)
print("Test samples:", x_test.shape)


# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0



# -----------------------------
# KNN CLASSIFIER
# -----------------------------

print("\nTraining KNN classifier...")


# Flatten images
x_train_flat = x_train.reshape(
    x_train.shape[0],
    -1
)

x_test_flat = x_test.reshape(
    x_test.shape[0],
    -1
)


# Use subset for faster execution
x_train_subset = x_train_flat[:5000]
y_train_subset = y_train[:5000]

x_test_subset = x_test_flat[:1000]
y_test_subset = y_test[:1000]


knn = KNeighborsClassifier(
    n_neighbors=5
)


knn.fit(
    x_train_subset,
    y_train_subset
)


y_pred_knn = knn.predict(
    x_test_subset
)


print("\n--- KNN Results ---")

print(
    "Accuracy:",
    accuracy_score(
        y_test_subset,
        y_pred_knn
    )
)


print(
    classification_report(
        y_test_subset,
        y_pred_knn
    )
)



# -----------------------------
# 3 LAYER NEURAL NETWORK
# -----------------------------

print("\nTraining 3-Layer Neural Network...")


# One-hot encoding
y_train_cat = to_categorical(
    y_train,
    10
)

y_test_cat = to_categorical(
    y_test,
    10
)



model = Sequential([

    Flatten(
        input_shape=(32, 32, 3)
    ),

    Dense(
        512,
        activation="relu"
    ),

    Dropout(
        0.3
    ),

    Dense(
        256,
        activation="relu"
    ),

    Dense(
        10,
        activation="softmax"
    )

])



model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)



# Train model

history = model.fit(
    x_train,
    y_train_cat,
    epochs=10,
    batch_size=128,
    validation_data=(
        x_test,
        y_test_cat
    )
)



# Evaluation

loss, accuracy = model.evaluate(
    x_test,
    y_test_cat
)


print("\n--- Neural Network Results ---")

print(
    f"Test Accuracy: {accuracy:.4f}"
)