import pandas as pd
import numpy as np

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BayesianEstimator
from pgmpy.inference import VariableElimination

from sklearn.preprocessing import LabelEncoder

import networkx as nx
import matplotlib.pyplot as plt



# Load dataset

df = pd.read_csv("heart.csv")


print("Dataset loaded successfully!\n")

print(df.head())



# Data preprocessing

label_encoders = {}


for column in df.columns:

    if df[column].dtype == "object":

        le = LabelEncoder()

        df[column] = le.fit_transform(
            df[column].astype(str)
        )

        label_encoders[column] = le



# Handle missing values

df = df.fillna(
    df.median(numeric_only=True)
)


print(
    "\nData preprocessing completed."
)

print(
    "Dataset shape:",
    df.shape
)



# Learn Bayesian Network structure

print(
    "\nLearning Bayesian Network structure..."
)


hc = HillClimbSearch(
    df
)


best_model = hc.estimate()


print(
    "Structure learned successfully!"
)


print(
    "Edges in learned network:",
    best_model.edges()
)



# Create Bayesian Network

model = DiscreteBayesianNetwork(
    best_model.edges()
)



# Train model

model.fit(
    df,
    estimator=BayesianEstimator,
    prior_type="BDeu"
)


print(
    "\nBayesian Network trained successfully!"
)



# Visualize Network

plt.figure(
    figsize=(10, 6)
)


G = nx.DiGraph()


G.add_edges_from(
    model.edges()
)


pos = nx.spring_layout(
    G,
    seed=42
)


nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=3000,
    node_color="skyblue",
    font_size=12,
    font_weight="bold",
    arrowsize=20
)


plt.title(
    "Bayesian Network Structure for Heart Disease"
)


plt.show()



# Inference

inference = VariableElimination(
    model
)



# Find target column

target_col = None


for col in df.columns:

    if col.lower() in [
        "target",
        "disease",
        "output"
    ]:

        target_col = col

        break



if target_col:


    print(
        f"\nPerforming heart disease diagnosis for column: {target_col}"
    )


    # Select random patient

    query_example = (
        df.sample(1)
        .iloc[0]
        .to_dict()
    )


    print(
        "\nExample patient data:"
    )

    print(
        query_example
    )


    # Remove target from evidence

    valid_evidence = {
        k: v
        for k, v in query_example.items()
        if k in model.nodes()
        and k != target_col
    }



    result = inference.map_query(
        [target_col],
        evidence=valid_evidence
    )


    print(
        f"\nPredicted Heart Disease Diagnosis → {target_col} = {result[target_col]}"
    )


else:


    print(
        "\nNo target column found."
        " Rename your target column as target/disease/output."
    )



print(
    "\nBayesian Network Heart Diagnosis completed successfully!"
)