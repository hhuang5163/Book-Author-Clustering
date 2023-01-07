import pandas as pd
import pyreadr
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def preprocess_data(data, key):
    '''
    Preprocesses data given the dictionary and key read from pyreadr

    Inputs:
        data: the dictionary of data as read by pyreadr
        key: which DataFrame to access and process
    
    Outputs:
        scaled_df: the processed dataset
    '''
    # Use data.keys() to identify there is only one key: 'authors'
    dfauthors = data[key]
    # Four authors: 'Austen', 'London', 'Milton', 'Shakespeare'
    scaler = preprocessing.StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(dfauthors), columns=dfauthors.columns)
    return dfauthors, scaled_df

def silhouette_plot(scaled_df, silscore, sample_silscores, labeled_data, cluster):
    '''
    Creates silhouette plot and plots the clustered data

    Inputs:
        scaled_df: the processed dataset
        silscore: mean silhouette coefficient of all samples
        sample_silscores: silhouette coefficient for each sample
        labeled_data: the clustered data done by the clustering method
        cluster: cluster size used in the clustering
    '''
    pca_authors = PCA(n_components=2).fit_transform(scaled_df) # For plotting axes
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_ylim([0, len(scaled_df) + (cluster + 1) * 10])
    y_counter = 10
    for i in range(cluster):
        silvalues_i = sample_silscores[labeled_data == i]
        silvalues_i.sort()
        color = cm.nipy_spectral(float(i) / cluster)
        ax1.fill_betweenx(np.arange(y_counter, y_counter + silvalues_i.shape[0]), 0, silvalues_i, facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_counter + 0.5 * silvalues_i.shape[0], str(i))
        y_counter += silvalues_i.shape[0] + 10
    
    ax1.set_title("Silhouette plot for " + str(cluster) + " clusters")
    ax1.set_xlabel("Silhouette Coefficient")
    ax1.set_ylabel("Clusters")
    ax1.set_xticks([-0.1, 0, 0.1, 0.2])
    ax1.set_yticks([])
    ax1.axvline(x=silscore, color="b", linestyle="-")

    colors = cm.nipy_spectral(labeled_data.astype(float) / cluster)
    ax2.scatter(pca_authors[:, 0], pca_authors[:, 1], c=colors)
    ax2.set_title("Clustered data (reduced to 2 PCs)")
    ax2.set_xlabel("Principal Component 1")
    ax2.set_ylabel("Principal Component 2")

def cluster_validation(mlmethod, clustersizes, scaled_df):
    '''
    Fits a model and clusters to every cluster size in clustersizes.
    Prints out silhouette scores and plots results to determine
    the optimal number of clusters.
    
    Inputs:
        mlmethod: machine learning clustering method to use. Currently 
                    can be 'kmeans' or 'hclustering'
                    'kmeans': K-Means
                    'hclustering': Hierarchical clustering
        clustersizes: list of all cluster sizes to test for the 
                        optimal number of clusters
        scaled_df: the preprocessed dataset

    '''
    for cluster in clustersizes:
        # Fits the ML model
        if mlmethod == 'kmeans':
            kmeans = KMeans(n_clusters=cluster)
            labeled = kmeans.fit_predict(scaled_df)
        elif mlmethod == 'hclustering':
            linkagetype = 'ward'
            distmetric = 'euclidean'
            aggclustering = AgglomerativeClustering(n_clusters=cluster, compute_distances=True, affinity=distmetric, linkage=linkagetype)
            labeled = aggclustering.fit_predict(scaled_df)
        # Calculates how well the model fits with the silhouette score
        silscore = silhouette_score(scaled_df, labeled)
        print("For " + str(cluster) + " clusters, the average silhouette score is: " + str(silscore))
        sample_silscores = silhouette_samples(scaled_df, labeled)
        silhouette_plot(scaled_df, silscore, sample_silscores, labeled, cluster)
    plt.show()


data = pyreadr.read_r('authors.rda')
dfauthors, scaled_df = preprocess_data(data, 'authors')
clustersizes = [2, 3, 4, 5, 6]
cluster_validation('kmeans', clustersizes, scaled_df)
cluster_validation('hclustering', clustersizes, scaled_df)

    