import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from sklearn.feature_extraction.text import CountVectorizer



# Load spam dataset

df = pd.read_csv(
    "spam.csv",
    encoding="latin-1"
)


# Select required columns

df = df[["v1", "v2"]]


# Rename columns

df.columns = [
    "label",
    "message"
]



# Convert labels

df["label"] = df["label"].map(
    {
        "ham": 0,
        "spam": 1
    }
)



print("\nSample Dataset:")

print(
    df.head()
)


print("\nColumns:")

print(
    df.columns
)



# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.8,
    random_state=42
)



# Text vectorization

vectorizer = CountVectorizer()


X_train_vec = vectorizer.fit_transform(
    X_train
)


X_test_vec = vectorizer.transform(
    X_test
)



# Train Naive Bayes classifier

model = MultinomialNB()


model.fit(
    X_train_vec,
    y_train
)



# Prediction

y_pred = model.predict(
    X_test_vec
)



# Evaluation

accuracy = accuracy_score(
    y_test,
    y_pred
)


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


conf_matrix = confusion_matrix(
    y_test,
    y_pred
)



# Results

print("\n--- Naive Bayes Document Classifier Results ---")

print(
    "Accuracy:",
    accuracy
)


print(
    "Precision:",
    precision
)


print(
    "Recall:",
    recall
)


print(
    "Confusion Matrix:\n",
    conf_matrix
)