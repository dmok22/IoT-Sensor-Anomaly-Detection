import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


# -------------------------------------------------------------
# STEP 1: Load and Inspect the Dataset
# -------------------------------------------------------------
def load_air_quality_data(csv_path):
    """
    Load the Air Quality dataset from a local CSV file.
    Download 'AirQualityUCI.csv' from the UCI ML Repository and place it in the same directory.
    """
    df = pd.read_csv(csv_path, sep=";", decimal=",", na_values=-200, low_memory=False)
    return df


# -------------------------------------------------------------
# STEP 2: Data Preprocessing
# -------------------------------------------------------------
def preprocess_data(df):
    """
    Preprocess the Air Quality data:
    - Drop columns that are mostly empty or not relevant (e.g. Date, Time if not needed).
    - Handle missing values.
    - Scale features.
    - Return the preprocessed NumPy array of features.
    """
    # The last two columns are empty (repeatedly) in the official dataset, so let's remove them
    if "Unnamed: 15" in df.columns:
        df.drop("Unnamed: 15", axis=1, inplace=True)
    if "Unnamed: 16" in df.columns:
        df.drop("Unnamed: 16", axis=1, inplace=True)

    # Drop rows with excessive NaN or missing data
    df.dropna(axis=0, how="any", inplace=True)

    # (Optional) Drop date and time columns if you only want to cluster based on sensor values
    if "Date" in df.columns:
        df.drop("Date", axis=1, inplace=True)
    if "Time" in df.columns:
        df.drop("Time", axis=1, inplace=True)

    # Convert DataFrame to numeric
    # Some columns might still be objects; force them to numeric if needed
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(axis=0, how="any", inplace=True)

    # Now scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    return data_scaled, df.columns.tolist()


# -------------------------------------------------------------
# STEP 3: Clustering with DBSCAN
# -------------------------------------------------------------
def cluster_dbscan(data, eps=0.9, min_samples=10):
    """
    Apply DBSCAN to the preprocessed data.
    Return the cluster labels and the DBSCAN model.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    return labels, dbscan


# -------------------------------------------------------------
# STEP 4: Compare with Other Methods
# -------------------------------------------------------------


def cluster_kmeans(data, n_clusters=5):
    """
    Apply K-Means clustering and return labels and model.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels, kmeans


def cluster_gmm(data, n_components=5):
    """
    Apply Gaussian Mixture Model clustering and return labels and model.
    """
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data)
    labels = gmm.predict(data)
    return labels, gmm


def detect_isolation_forest(data, contamination=0.05):
    """
    Apply Isolation Forest for outlier detection.
    This returns +1 for inliers, -1 for outliers.
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_labels = iso_forest.fit_predict(data)
    return iso_labels, iso_forest


# -------------------------------------------------------------
# STEP 5: Evaluation and Visualization
# -------------------------------------------------------------
def evaluate_clustering(data, labels, method_name="Method"):
    """
    Compute silhouette score and Davies-Bouldin score for a given clustering.
    Ignore labels where everything is in one cluster or -1 to avoid error in silhouette computation.
    """
    unique_labels = np.unique(labels)
    # If we have fewer than 2 clusters, skip silhouette
    if len(unique_labels) < 2:
        print(
            f"{method_name}: Not enough clusters for silhouette. All data might be in one cluster or outlier cluster."
        )
        return

    # Silhouette Score
    sil_score = silhouette_score(data, labels)

    # Davies-Bouldin Score
    db_score = davies_bouldin_score(data, labels)

    print(
        f"{method_name}: Silhouette Score = {sil_score:.4f}, Davies-Bouldin Score = {db_score:.4f}"
    )


def visualize_clusters(data, labels, title="Clustering Visualization"):
    """
    Simple 2D visualization using the first two principal components
    or just the first two features for demonstration.
    """
    # If data has high dimensionality, we can do a quick PCA to 2D for visualization
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)

    plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, alpha=0.7)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Cluster / Outlier Label")
    plt.show()


# -------------------------------------------------------------
# MAIN EXECUTION BLOCK
# -------------------------------------------------------------
if __name__ == "__main__":
    # 1) Load data
    csv_file = "AirQualityUCI.csv"  # Update with your path
    df = load_air_quality_data(csv_file)

    # 2) Preprocess data
    data_scaled, feature_names = preprocess_data(df)
    print("Data Shape after cleaning:", data_scaled.shape)
    print("Feature names used:", feature_names)

    # 3) DBSCAN
    dbscan_labels, dbscan_model = cluster_dbscan(data_scaled, eps=0.9, min_samples=10)
    print("DBSCAN unique cluster labels:", np.unique(dbscan_labels))
    evaluate_clustering(data_scaled, dbscan_labels, method_name="DBSCAN")
    visualize_clusters(data_scaled, dbscan_labels, title="DBSCAN Clusters")

    # 4) K-Means
    kmeans_labels, kmeans_model = cluster_kmeans(data_scaled, n_clusters=5)
    print("K-Means unique cluster labels:", np.unique(kmeans_labels))
    evaluate_clustering(data_scaled, kmeans_labels, method_name="K-Means (5 clusters)")
    visualize_clusters(data_scaled, kmeans_labels, title="K-Means Clusters")

    # 5) GMM
    gmm_labels, gmm_model = cluster_gmm(data_scaled, n_components=5)
    print("GMM unique cluster labels:", np.unique(gmm_labels))
    evaluate_clustering(
        data_scaled, gmm_labels, method_name="Gaussian Mixture (5 components)"
    )
    visualize_clusters(data_scaled, gmm_labels, title="Gaussian Mixture Model Clusters")

    # 6) Isolation Forest for Outlier Detection
    iso_labels, iso_forest = detect_isolation_forest(data_scaled, contamination=0.05)
    # iso_labels are +1 (inlier) or -1 (outlier)
    # We'll do a small trick for silhouette: label outliers as one "cluster" and inliers as another
    # That means we can transform the labels to 0 and 1 to evaluate if needed
    iso_numeric_labels = np.where(iso_labels == -1, 1, 0)

    print("Isolation Forest unique labels (inlier/outlier):", np.unique(iso_labels))
    evaluate_clustering(data_scaled, iso_numeric_labels, method_name="Isolation Forest")
    visualize_clusters(
        data_scaled, iso_numeric_labels, title="Isolation Forest Outliers"
    )

    # Observing the results in the console output helps you decide which method
    # yields better cluster separation or outlier detection performance.
