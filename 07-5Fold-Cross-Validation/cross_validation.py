import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.tree import DecisionTreeClassifier


# Load dataset
data = pd.read_csv("iris.csv")

print("Dataset loaded successfully!")
print(data.head())


# Identify target column
target_col = "species" if "species" in data.columns else data.columns[-1]


# Split features and labels
X = data.drop(target_col, axis=1)
y = data[target_col]


# 5-Fold Cross Validation
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


accuracy_list = []
precision_list = []
recall_list = []
f1_list = []


fold = 1


for train_index, test_index in kf.split(X):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]


    # Decision Tree Classifier
    model = DecisionTreeClassifier(
        criterion="entropy",
        random_state=42
    )


    model.fit(
        X_train,
        y_train
    )


    y_pred = model.predict(X_test)


    # Metrics
    acc = accuracy_score(
        y_test,
        y_pred
    )

    prec = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    rec = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )


    accuracy_list.append(acc)
    precision_list.append(prec)
    recall_list.append(rec)
    f1_list.append(f1)


    print(f"\nFold {fold}")
    print("----------------")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1 Score : {f1:.3f}")


    fold += 1



# Overall Performance
print("\n--- Overall Performance (Average of 5 Folds) ---")

print(
    f"Mean Accuracy : {sum(accuracy_list)/5:.3f}"
)

print(
    f"Mean Precision: {sum(precision_list)/5:.3f}"
)

print(
    f"Mean Recall   : {sum(recall_list)/5:.3f}"
)

print(
    f"Mean F1 Score : {sum(f1_list)/5:.3f}"
)