import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load Iris dataset
data = pd.read_csv(
    "https://raw.githubusercontent.com/ms-starryvoid/ML_Lab_DataSet/refs/heads/main/Lab_data/Iris.csv"
)

print("Sample Dataset:\n")
print(data.head())

print("\nDataset Information:")
print(data.info())


# Remove target column
if "Species" in data.columns:
    data = data.drop(columns=["Species"])


# Remove unnecessary ID column if present
if "User ID" in data.columns:
    data = data.drop(columns=["User ID"])


# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

kmeans_labels = kmeans.fit_predict(data)


# EM Algorithm using Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)

gmm_labels = gmm.fit_predict(data)


# Evaluation Metrics

sil_kmeans = silhouette_score(data, kmeans_labels)
sil_gmm = silhouette_score(data, gmm_labels)

db_kmeans = davies_bouldin_score(data, kmeans_labels)
db_gmm = davies_bouldin_score(data, gmm_labels)


print("\nPerformance Comparison")

print(f"KMeans Silhouette Score: {sil_kmeans:.3f}")
print(f"GMM Silhouette Score: {sil_gmm:.3f}")

print(f"KMeans Davies-Bouldin Score: {db_kmeans:.3f}")
print(f"GMM Davies-Bouldin Score: {db_gmm:.3f}")


# Visualization Function


def plot_clusters(data, labels, title):

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        x=data.iloc[:, 0],
        y=data.iloc[:, 1],
        hue=labels,
        palette="Set2",
        s=100,
        edgecolor="black",
    )

    plt.title(title)
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.legend(title="Cluster")
    plt.grid(True)

    plt.show()


# Plot Results

plot_clusters(data, kmeans_labels, "KMeans Clustering")


plot_clusters(data, gmm_labels, "Gaussian Mixture Model (EM) Clustering")
