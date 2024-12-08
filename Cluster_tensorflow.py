import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function to perform KMeans clustering using TensorFlow
def kmeans_tf(X, n_clusters=4, n_iterations=100):
    n_samples, n_features = X.shape
    X = tf.Variable(X, dtype=tf.float32)

    # Initialize centroids randomly
    indices = tf.random.shuffle(tf.range(n_samples))[:n_clusters]
    centroids = tf.Variable(tf.gather(X, indices))

    # Run KMeans algorithm
    for _ in range(n_iterations):
        # Calculate the distance between each point and the centroids
        distances = tf.norm(tf.expand_dims(X, 1) - centroids, axis=2)
        labels = tf.argmin(distances, axis=1)

        # Update centroids by averaging the points in each cluster
        for i in range(n_clusters):
            cluster_points = tf.boolean_mask(X, tf.equal(labels, i))
            if len(cluster_points) > 0:
                centroids[i].assign(tf.reduce_mean(cluster_points, axis=0))

    return labels.numpy(), centroids.numpy()

# Reading the data from the CSV file
df = pd.read_csv('Mall_Customers.csv')

# 1. Segmentation using Age and Spending Score
X1 = df[['Age', 'Spending Score (1-100)']].values


# Performing KMeans clustering with the optimal number of clusters (based on elbow method)
optimal_clusters = 4  # Change this value based on the elbow plot
labels1, centroids1 = kmeans_tf(X1, n_clusters=optimal_clusters)

# Visualizing the clustering result
plt.figure(1, figsize=(15, 7))
plt.scatter(X1[:, 0], X1[:, 1], c=labels1, s=200, cmap='Pastel2')
plt.scatter(centroids1[:, 0], centroids1[:, 1], s=300, c='red', alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('Clustering by Age and Spending Score')
plt.show()

# 2. Segmentation using Annual Income and Spending Score
X2 = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Performing KMeans clustering with optimal clusters
labels2, centroids2 = kmeans_tf(X2, n_clusters=5)

# Visualizing the clustering result
plt.figure(2, figsize=(15, 7))
plt.scatter(X2[:, 0], X2[:, 1], c=labels2, s=200, cmap='Pastel2')
plt.scatter(centroids2[:, 0], centroids2[:, 1], s=300, c='red', alpha=0.5)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Clustering by Annual Income and Spending Score')
plt.show()

# 3. Segmentation using Age, Annual Income and Spending Score
X3 = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

# Performing KMeans clustering with optimal clusters
labels3, centroids3 = kmeans_tf(X3, n_clusters=6)

# Visualizing the 3D clustering result
fig = plt.figure(3, figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=labels3, s=200, cmap='Pastel2')
ax.scatter(centroids3[:, 0], centroids3[:, 1], centroids3[:, 2], s=300, c='red', alpha=0.5)
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
ax.set_title('3D Clustering by Age, Annual Income and Spending Score')
plt.show()
