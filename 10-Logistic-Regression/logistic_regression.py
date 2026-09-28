import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)


# Load dataset

df = pd.read_csv("diabetes.csv")


print("Dataset loaded successfully!")

print(df.head())



# Split features and target

X = df.drop(
    columns="Outcome"
)

y = df["Outcome"]



# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)



# Logistic Regression Model

model = LogisticRegression(
    max_iter=1000
)



# Training

model.fit(
    X_train,
    y_train
)



# Prediction

y_pred = model.predict(
    X_test
)



# Evaluation metrics

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)


recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)


f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


accuracy = accuracy_score(
    y_test,
    y_pred
)



# Display results

print("\n--- Logistic Regression Performance ---")

print(
    "Precision :",
    round(precision, 4)
)

print(
    "Recall    :",
    round(recall, 4)
)

print(
    "F1 Score  :",
    round(f1, 4)
)

print(
    "Accuracy  :",
    round(accuracy, 4)
)