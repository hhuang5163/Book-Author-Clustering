import pandas as pd
import pyreadr
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import SpectralEmbedding
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

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

def author_accuracy(cluster_list):
    '''
    Computes fraction of correctly classified book authors
    given book chapters

    Inputs:
        cluster_list: list of classifications of authors
    
    Outputs:
        Fraction of correctly classified authors
    '''
    correctcount = 0
    authorcounts = [317, 296, 173, 55]
    startidx = 0
    for k in range(4):
        countsdict = {0: 0, 1: 0, 2: 0, 3: 0}
        for i in range(startidx, authorcounts[k] + startidx):
            countsdict[cluster_list[i]] += 1
        correctcount += max(countsdict.values())
        startidx += authorcounts[k]
    return correctcount / sum(authorcounts)

def plot_dendrogram(model, **kwargs):
    '''
    Plots the dendrogram for hierarchical clustering

    Inputs:
        model: the scikit-learn model used to perform 
                hierarchical clustering

    Outputs:
        Plot of the dendrogram used in hierarchical 
        clustering
    '''
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    dendrogram(linkage_matrix, **kwargs)

def kmeans(scaled_df):
    '''
    Clusters using k-means method and plots the clustering

    Inputs:
        scaled_df: the preprocessed DataFrame
    
    Outputs:
        Plot of clustering based on k-means
    '''
    # Fits kmeans model
    kmeans = KMeans(n_clusters=4).fit_predict(scaled_df)
    accuracy = author_accuracy(kmeans)
    # Plots on PCA first two components the K means clustering results
    pca_authors = PCA(n_components=2).fit_transform(scaled_df)
    # Plots the k-means clustering results
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(pca_authors[:, 0], pca_authors[:, 1], c=kmeans)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('KMeans Clustering (4 Clusters) with Accuracy: ' + str(accuracy))
    fig.savefig('Clustering Accuracy/ClusteringKMeans4.png')
    plt.show()

def hierarchical_clustering(scaled_df, linkagetype, distmetric):
    '''
    Clusters using hierarchical clustering and plots the dendrogram

    Inputs:
        scaled_df: the preprocessed DataFrame
        linkagetype: the type of linkage criterion to use between 
                        sets of observations
        distmetric: metric to use when calculating distance between 
                        two data points
    
    Outputs:
        Dendrogram used to cluster based on hierarchical clustering
    '''
    # Fits the model
    aggclustering = AgglomerativeClustering(n_clusters=4, compute_distances=True, affinity=distmetric, linkage=linkagetype)
    labels = aggclustering.fit_predict(scaled_df)
    aggclustering = aggclustering.fit(scaled_df)
    # Plots and saves the dendrogram
    plot_dendrogram(aggclustering)
    accuracy = author_accuracy(labels)
    plt.title("Dendrogram (K=4) using linkage " + linkagetype + " and " + distmetric + " distance with accuracy: " + str(accuracy))
    fig = plt.gcf()
    fig.savefig(f'Clustering Accuracy/HC4{linkagetype}{distmetric}.png')
    plt.show()

def spectral_embedding(scaled_df):
    '''
    Clusters using spectral embedding and plots the clustering

    Inputs:
        scaled_df: the preprocessed DataFrame
    
    Outputs:
        Plot of clustering based on spectral embedding
    '''
    # Fits the spectral embedding model
    se_authors = SpectralEmbedding().fit_transform(scaled_df)
    se_df = pd.DataFrame(data = se_authors[:, 0:2], columns = ['Component 1', 'Component 2'])
    se_df = pd.concat([se_df, pd.DataFrame(data = dfauthors.index.values, columns = ['target'])], axis = 1)
    # Plots the spectral embedding
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title('Spectral Embedding for Book Chapters')
    targets = ['Austen', 'London', 'Milton', 'Shakespeare']
    colors = ['r', 'g', 'b', 'y']
    for target, color in zip(targets,colors):
        target_idxs = se_df['target'] == target
        ax.scatter(se_df.loc[target_idxs, 'Component 1']
                , se_df.loc[target_idxs, 'Component 2']
                , c = color
                , s = 20)
    ax.legend(targets)
    fig.savefig('Clustering Accuracy/SpectralEmbedding4.png')
    plt.show()


data = pyreadr.read_r('authors.rda')
dfauthors, scaled_df = preprocess_data(data, 'authors')
kmeans(scaled_df)
hierarchical_clustering(scaled_df, linkagetype='ward', distmetric='euclidean')
spectral_embedding(scaled_df)
