import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("iris.csv")


# Split features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42
)


# Feature scaling (important for KNN)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Create KNN model
k = 3

knn = KNeighborsClassifier(
    n_neighbors=k
)


# Train model
knn.fit(
    X_train,
    y_train
)


# Prediction
y_pred = knn.predict(X_test)


# Store predictions
correct_predictions = []
wrong_predictions = []


for i in range(len(y_test)):

    if y_test.iloc[i] == y_pred[i]:

        correct_predictions.append(
            (
                X_test[i],
                y_test.iloc[i],
                y_pred[i]
            )
        )

    else:

        wrong_predictions.append(
            (
                X_test[i],
                y_test.iloc[i],
                y_pred[i]
            )
        )


# Display correct predictions
print("Correct Predictions:\n")

for prediction in correct_predictions:

    print(
        f"Features: {prediction[0]} | "
        f"True Label: {prediction[1]} | "
        f"Predicted Label: {prediction[2]}"
    )


# Display wrong predictions
print("\nWrong Predictions:\n")

for prediction in wrong_predictions:

    print(
        f"Features: {prediction[0]} | "
        f"True Label: {prediction[1]} | "
        f"Predicted Label: {prediction[2]}"
    )


# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)