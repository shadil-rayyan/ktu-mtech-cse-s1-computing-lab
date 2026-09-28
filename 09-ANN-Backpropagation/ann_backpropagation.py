import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score


# Load Iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

data = pd.read_csv(url)


# Feature and label separation
X = data.drop("species", axis=1).values

y = LabelEncoder().fit_transform(
    data["species"]
)


# Feature scaling
scaler = StandardScaler()

X = scaler.fit_transform(X)



# One-hot encoding
y_encoded = np.zeros(
    (y.size, y.max() + 1)
)

y_encoded[
    np.arange(y.size),
    y
] = 1



# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)



# Neural network parameters

input_neurons = X_train.shape[1]

hidden_neurons = 6

output_neurons = y_train.shape[1]


learning_rate = 0.05

epochs = 5000



# Initialize weights

np.random.seed(42)


W1 = np.random.randn(
    input_neurons,
    hidden_neurons
)

b1 = np.zeros(
    (1, hidden_neurons)
)


W2 = np.random.randn(
    hidden_neurons,
    output_neurons
)

b2 = np.zeros(
    (1, output_neurons)
)



# Activation function

def sigmoid(x):

    return 1 / (1 + np.exp(-x))



def sigmoid_derivative(x):

    return x * (1 - x)



# Training using Backpropagation

for epoch in range(epochs):


    # Forward propagation

    z1 = np.dot(X_train, W1) + b1

    a1 = sigmoid(z1)


    z2 = np.dot(a1, W2) + b2

    a2 = sigmoid(z2)



    # Loss calculation

    loss = np.mean(
        (y_train - a2) ** 2
    )



    # Backpropagation

    error_output = y_train - a2

    delta_output = (
        error_output *
        sigmoid_derivative(a2)
    )


    error_hidden = delta_output.dot(
        W2.T
    )


    delta_hidden = (
        error_hidden *
        sigmoid_derivative(a1)
    )



    # Update weights

    W2 += (
        a1.T.dot(delta_output)
        * learning_rate
    )


    b2 += (
        np.sum(
            delta_output,
            axis=0,
            keepdims=True
        )
        * learning_rate
    )


    W1 += (
        X_train.T.dot(delta_hidden)
        * learning_rate
    )


    b1 += (
        np.sum(
            delta_hidden,
            axis=0,
            keepdims=True
        )
        * learning_rate
    )



    if epoch % 500 == 0:

        print(
            f"Epoch {epoch}, Loss: {loss:.4f}"
        )



# Testing

z1_test = np.dot(
    X_test,
    W1
) + b1


a1_test = sigmoid(
    z1_test
)


z2_test = np.dot(
    a1_test,
    W2
) + b2


a2_test = sigmoid(
    z2_test
)



# Prediction

y_pred = np.argmax(
    a2_test,
    axis=1
)


y_true = np.argmax(
    y_test,
    axis=1
)



# Accuracy

accuracy = accuracy_score(
    y_true,
    y_pred
)


print(
    "\nTest Accuracy:",
    round(accuracy * 100, 2),
    "%"
)