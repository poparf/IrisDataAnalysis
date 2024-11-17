from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from main import df, feature_names, X, X_scaled, y
from pca import X_pca

# Find optimal number of clusters using elbow method

inertias = [] # a measure of how well the clusters are formed
# defined as sum of squared distances between each point and the centroid of the cluster
# it belong to. The smaller the inertia the denser the cluster.

# Silhouette score is a measure of how similar a point is to its own cluster
# compared to other clusters. It ranges from -1 to 1.
silhouette_scores = []

total_variation = sum((X_scaled - X_scaled.mean(axis=0))**2)/len(X_scaled)
reduction_of_variation = []
K = range(2, 8)

for k in K:
    # random_state is like a seed for the random number generator
    kmeans = KMeans(n_clusters=k, random_state=37)
    # fit() computes k-means clustering
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    explained_variation = total_variation - kmeans.inertia_
    reduction_of_variation.append(explained_variation / total_variation)

# Perform K-means clustering with optimal k=3 
# (known for iris dataset since there are 3 species)
kmeans = KMeans(n_clusters=3, random_state=42)
# fit_predict() computes cluster centers and predicts cluster index for each sample
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to DataFrame
df['Cluster'] = clusters # In the case of the dataset the cluster wil be representative
# with the species of the flower. 
# So the cluster 0 will be Setosa, 
# cluster 1 will be Versicolor and cluster 2 will be Virginica.

# Create visualization functions
# Elbow curve shows the optimal number of clusters
def plot_elbow_curve():
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    
    plt.subplot(1, 2, 2)
    plt.plot(K, silhouette_scores, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Method')
    
    plt.tight_layout()
    plt.savefig('../output/clustering/elbow_curve.png')
    plt.close()
        
    # Plot reduction of variation against cluster number k
    plt.figure(figsize=(10, 5))
    plt.plot(K, reduction_of_variation, 'gx-')
    plt.xlabel('k')
    plt.ylabel('Reduction of Variation')
    plt.title('Reduction of Variation vs. Number of Clusters')
    plt.savefig('../output/clustering/reduction_of_variation.png')
    plt.close()

def plot_3d_clusters():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X_scaled[:, 0], 
                        X_scaled[:, 1], 
                        X_scaled[:, 2],
                        c=clusters,
                        cmap='viridis')
    
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(feature_names[2])
    plt.title('3D Cluster Visualization')
    plt.colorbar(scatter)
    plt.savefig('../output/clustering/3d_clusters.png')
    plt.close()
    
def plot_scatter_principal_components_with_cluster():
    kmeans = KMeans(n_clusters=3, random_state=42)
    # fit_predict() computes cluster centers and predicts cluster index for each sample
    clusters = kmeans.fit_predict(X_pca)
    # K means doesn't guarantee the ordering of clusters
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, 
                        cmap='viridis')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Iris Dataset - First Two Principal Components')
    # Create a legend with custom labels
    handles, _ = scatter.legend_elements()
    plt.legend(handles, ['Cluster 0', 'Cluster 1', 'Cluster 2'])
    plt.savefig('../output/clustering/pca_clusters.png')
    plt.close()
    
def analyze_clusters():
    # Create summary statistics for each cluster
    cluster_summary = df.groupby('Cluster').agg({
        feature_names[0]: ['mean', 'std', 'count'],
        feature_names[1]: ['mean', 'std'],
        feature_names[2]: ['mean', 'std'],
        feature_names[3]: ['mean', 'std']
    }).round(2)
    
    return cluster_summary

# Run the analysis
plot_elbow_curve()
plot_3d_clusters()
plot_scatter_principal_components_with_cluster()

cluster_summary = analyze_clusters()
print("\nCluster Summary Statistics:")
print(cluster_summary)


with open('../output/clustering/clustering_analysis_results.txt', 'w') as f:
    f.write("Clustering Analysis Results for Iris Dataset\n")
    f.write(cluster_summary.to_string())