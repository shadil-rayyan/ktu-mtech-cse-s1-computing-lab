import pandas as pd
import numpy as np


# Load dataset

data = pd.read_csv("naiv (2).csv")


print("First 5 rows of the dataset:")
print(data.head())



# Split features and labels

X = data.iloc[:, :-1].values

y = data.iloc[:, -1].values



# Shuffle dataset

np.random.seed(42)

indices = np.arange(len(X))

np.random.shuffle(indices)



# Train-test split (80-20)

split = int(0.8 * len(X))


train_idx = indices[:split]

test_idx = indices[split:]


X_train = X[train_idx]

y_train = y[train_idx]


X_test = X[test_idx]

y_test = y[test_idx]



# Gaussian Naive Bayes Implementation

class GaussianNB_FromScratch:


    def fit(self, X, y):

        self.classes = np.unique(y)

        self.mean = {}

        self.var = {}

        self.priors = {}


        for c in self.classes:

            X_c = X[y == c]


            self.mean[c] = np.mean(
                X_c,
                axis=0
            )


            self.var[c] = np.var(
                X_c,
                axis=0
            ) + 1e-6


            self.priors[c] = (
                X_c.shape[0] /
                X.shape[0]
            )



    # Gaussian probability density function

    def gaussian_pdf(self, class_idx, x):

        mean = self.mean[class_idx]

        var = self.var[class_idx]


        numerator = np.exp(
            -((x - mean) ** 2) /
            (2 * var)
        )


        denominator = np.sqrt(
            2 * np.pi * var
        )


        return numerator / denominator



    # Predict single sample

    def predict_one(self, x):

        posteriors = {}


        for c in self.classes:


            prior = np.log(
                self.priors[c]
            )


            conditional = np.sum(
                np.log(
                    self.gaussian_pdf(c, x)
                )
            )


            posteriors[c] = (
                prior +
                conditional
            )


        return max(
            posteriors,
            key=posteriors.get
        )



    # Predict multiple samples

    def predict(self, X):

        return np.array(
            [
                self.predict_one(x)
                for x in X
            ]
        )



# Create and train model

model = GaussianNB_FromScratch()


model.fit(
    X_train,
    y_train
)



# Prediction

y_pred = model.predict(
    X_test
)



# Accuracy

accuracy = np.mean(
    y_pred == y_test
)



print(
    f"\nAccuracy of Naive Bayes classifier: {accuracy * 100:.2f}%"
)



# Prediction comparison

comparison = pd.DataFrame(
    {
        "Actual": y_test,
        "Predicted": y_pred
    }
)


print("\nSample Predictions:")

print(
    comparison.head()
)