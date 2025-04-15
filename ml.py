import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from tabulate import (
    tabulate,
)  # For nice feature table printing (install via pip if needed)


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
    - Remove extraneous columns (e.g., 'Unnamed: 15', 'Unnamed: 16').
    - Drop rows with missing values rather than imputing to avoid introducing bias
      (unsupervised models depend on natural density and distance metrics).
    - Drop 'Date' and 'Time' columns if clustering is only based on sensor values.
    - Convert columns to numeric and drop any remaining NaN values.
    - Scale features with StandardScaler (distance-based methods benefit from scaling).
    - Return the scaled data and the list of final feature names.
    """
    # Remove extraneous columns that are empty
    for col in ["Unnamed: 15", "Unnamed: 16"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Drop rows with missing data
    df.dropna(axis=0, how="any", inplace=True)

    # Drop Date and Time if present (we assume clustering based solely on sensor values)
    for col in ["Date", "Time"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Force conversion to numeric (any non-numeric columns will become NaN and then dropped)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(axis=0, how="any", inplace=True)

    # Print summary table of final features
    feature_names = df.columns.tolist()
    # Here you could add original units if available; we use placeholder units
    feature_table = [
        [feat, "N/A"] for feat in feature_names
    ]  # Replace "N/A" with unit if known
    print("\nFinal Features and Units:")
    print(
        tabulate(
            feature_table, headers=["Feature Name", "Original Unit"], tablefmt="github"
        )
    )

    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    return data_scaled, feature_names


# -------------------------------------------------------------
# STEP 3: Clustering with DBSCAN (with Parameter Sensitivity Testing)
# -------------------------------------------------------------
def cluster_dbscan(data, eps=0.9, min_samples=10):
    """
    Apply DBSCAN to the preprocessed data.
    Return the cluster labels and the DBSCAN model.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    return labels, dbscan


def tune_dbscan_eps(data, eps_values, min_samples=10):
    """
    Evaluate DBSCAN with different eps values and print silhouette scores.
    This helps in selecting an appropriate eps using a k-distance plot approach.
    """
    for eps in eps_values:
        labels, _ = cluster_dbscan(data, eps=eps, min_samples=min_samples)
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            print(
                f"eps={eps}: Not enough clusters for evaluation (labels: {unique_labels})."
            )
        else:
            sil_score = silhouette_score(data, labels)
            db_score = davies_bouldin_score(data, labels)
            print(
                f"eps={eps:.2f}: Silhouette Score = {sil_score:.4f}, Davies-Bouldin Score = {db_score:.4f}, Clusters = {len(unique_labels)}"
            )


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
    gmm = GaussianMixture(
        n_components=n_components, random_state=42, covariance_type="full"
    )
    gmm.fit(data)
    labels = gmm.predict(data)
    return labels, gmm


def detect_isolation_forest(data, contamination=0.05):
    """
    Apply Isolation Forest for outlier detection.
    Returns +1 for inliers, -1 for outliers.
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
    If fewer than 2 clusters are found, print a warning.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print(
            f"{method_name}: Not enough clusters for silhouette evaluation. (Found labels: {unique_labels})"
        )
        return None, None
    sil_score = silhouette_score(data, labels)
    db_score = davies_bouldin_score(data, labels)
    print(
        f"{method_name}: Silhouette Score = {sil_score:.4f}, Davies-Bouldin Score = {db_score:.4f}"
    )
    return sil_score, db_score


def visualize_clusters(data, labels, title="Clustering Visualization"):
    """
    Visualize clusters by reducing the data to 2 dimensions using PCA.
    """
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)
    plt.figure()
    scatter = plt.scatter(
        reduced[:, 0], reduced[:, 1], c=labels, alpha=0.7, cmap="viridis"
    )
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Cluster / Outlier Label")
    plt.show()


# -------------------------------------------------------------
# MAIN EXECUTION BLOCK
# -------------------------------------------------------------
if __name__ == "__main__":
    # 1) Load data
    csv_file = "AirQualityUCI.csv"  # Update with your correct path if needed
    df = load_air_quality_data(csv_file)

    # 2) Preprocess data
    data_scaled, feature_names = preprocess_data(df)
    print("Data Shape after cleaning:", data_scaled.shape)
    print("Feature names used:", feature_names)

    # 3) DBSCAN
    print("\n--- DBSCAN Parameter Tuning (eps values) ---")
    eps_values = np.linspace(0.85, 0.95, 5)
    tune_dbscan_eps(data_scaled, eps_values, min_samples=10)

    # Using chosen parameters eps=0.9 and min_samples=10 for further analysis
    dbscan_labels, dbscan_model = cluster_dbscan(data_scaled, eps=0.9, min_samples=10)
    print("DBSCAN unique cluster labels:", np.unique(dbscan_labels))
    evaluate_clustering(data_scaled, dbscan_labels, method_name="DBSCAN")
    visualize_clusters(data_scaled, dbscan_labels, title="DBSCAN Clusters")

    # 4) K-Means Clustering
    kmeans_labels, kmeans_model = cluster_kmeans(data_scaled, n_clusters=5)
    print("\nK-Means unique cluster labels:", np.unique(kmeans_labels))
    evaluate_clustering(data_scaled, kmeans_labels, method_name="K-Means (5 clusters)")
    visualize_clusters(data_scaled, kmeans_labels, title="K-Means Clusters")

    # 5) Gaussian Mixture Models (GMM)
    gmm_labels, gmm_model = cluster_gmm(data_scaled, n_components=5)
    print("\nGMM unique cluster labels:", np.unique(gmm_labels))
    evaluate_clustering(
        data_scaled, gmm_labels, method_name="Gaussian Mixture (5 components)"
    )
    visualize_clusters(data_scaled, gmm_labels, title="Gaussian Mixture Model Clusters")

    # 6) Isolation Forest for Outlier Detection
    iso_labels, iso_forest = detect_isolation_forest(data_scaled, contamination=0.05)
    # Convert isolation forest labels (+1 for inliers, -1 for outliers) to binary clusters: 0 for inliers, 1 for outliers.
    iso_numeric_labels = np.where(iso_labels == -1, 1, 0)
    print("\nIsolation Forest unique labels (inlier/outlier):", np.unique(iso_labels))
    evaluate_clustering(data_scaled, iso_numeric_labels, method_name="Isolation Forest")
    visualize_clusters(
        data_scaled, iso_numeric_labels, title="Isolation Forest Outliers"
    )

    # 7) Cross-Validation / Robustness Testing for K-Means and GMM
    # Here we run multiple iterations to get mean and std for silhouette scores.
    n_runs = 10
    kmeans_scores = []
    gmm_scores = []
    for seed in range(42, 42 + n_runs):
        kmeans = KMeans(n_clusters=5, random_state=seed)
        kmeans_labels_cv = kmeans.fit_predict(data_scaled)
        if len(np.unique(kmeans_labels_cv)) > 1:
            kmeans_scores.append(silhouette_score(data_scaled, kmeans_labels_cv))
        gmm = GaussianMixture(n_components=5, random_state=seed, covariance_type="full")
        gmm.fit(data_scaled)
        gmm_labels_cv = gmm.predict(data_scaled)
        if len(np.unique(gmm_labels_cv)) > 1:
            gmm_scores.append(silhouette_score(data_scaled, gmm_labels_cv))

    if kmeans_scores:
        print(
            "\nK-Means Silhouette Score: Mean = {:.4f}, Std = {:.4f}".format(
                np.mean(kmeans_scores), np.std(kmeans_scores)
            )
        )
    if gmm_scores:
        print(
            "GMM Silhouette Score: Mean = {:.4f}, Std = {:.4f}".format(
                np.mean(gmm_scores), np.std(gmm_scores)
            )
        )

    # For Isolation Forest, test robustness with contamination rate in range [0.03, 0.07]
    contamination_values = np.linspace(0.03, 0.07, 5)
    for cont in contamination_values:
        iso_labels_temp, _ = detect_isolation_forest(data_scaled, contamination=cont)
        iso_num_labels = np.where(iso_labels_temp == -1, 1, 0)
        unique, counts = np.unique(iso_num_labels, return_counts=True)
        print(
            f"Isolation Forest with contamination {cont:.2f}: Outlier Count = {dict(zip(unique, counts))}"
        )

    # End of main execution block.
