import pandas as pd
import pyreadr
import numpy as np
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import MDS, TSNE, SpectralEmbedding
from sklearn import preprocessing
from sklearn.neighbors import DistanceMetric
import matplotlib.pyplot as plt
import plotly.express as px

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
    return dfauthors.index.values, scaled_df

def plot_data(df, title):
    '''
    Plots the data from the DataFrame, specifically for the four authors

    Inputs:
        df: the DataFrame to be plotted
        title: the title of the plot

    Outputs:
        A plot illustrating the data after a dimension reduction technique 
        is applied to the data
    '''
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title(f'{title} for Book Chapters')
    targets = ['Austen', 'London', 'Milton', 'Shakespeare']
    colors = ['r', 'g', 'b', 'y']
    for target, color in zip(targets,colors):
        target_idxs = df['target'] == target
        ax.scatter(df.loc[target_idxs, 'Component 1']
                , df.loc[target_idxs, 'Component 2']
                , c = color
                , s = 20)
    ax.legend(targets)
    fig.savefig(f'Results Book Chapters/{title}.png')
    plt.show()

def pca_screeplot(scaled_df):
    '''
    Plots the screeplot for the fit PCA model

    Inputs:
        scaled_df: the preprocessed DataFrame
    
    Outputs:
        Screeplot of how much variance is explained by each principal component
    '''
    # Plots explained variance for book chapters
    pca_authors = PCA().fit(scaled_df)
    # print(pca_authors.explained_variance_ratio_)
    pc_vals = np.arange(pca_authors.n_components_) + 1
    plt.plot(pc_vals, pca_authors.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot for Book Chapters')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    fig = plt.gcf()
    fig.savefig('Results Book Chapters/PCAvariance.png')
    plt.show()

def pca_pcs(scaled_df, dfidx_vals):
    '''
    Plots the first two principal components to see if the data
    naturally clusters

    Inputs: 
        scaled_df: the preprocessed DataFrame
        dfidx_vals: corresponding author for each row in the DataFrame

    Outputs:
        Plots the data according to the first two principal components
    '''
    # Fits the PCA model
    pca_authors = PCA(n_components=2).fit_transform(scaled_df)
    pca_df = pd.DataFrame(data = pca_authors, columns = ['Component 1', 'Component 2'])
    pca_df = pd.concat([pca_df, pd.DataFrame(data = dfidx_vals, columns = ['target'])], axis = 1)

    # Plots the first two principal components
    plot_data(pca_df, 'PCA')

def nmf(scaled_df, dfidx_vals):
    '''
    Plots the first two dimensions after applying NMF to the data

    Inputs: 
        scaled_df: the preprocessed DataFrame
        dfidx_vals: corresponding author for each row in the DataFrame

    Outputs:
        Plots the data according to the first two dimensions
    '''
    # Scales data such that they are all non-negative
    scaled_df = preprocessing.MinMaxScaler().fit_transform(scaled_df)
    # Fits NMF model
    nmf_authors = NMF()
    nmf_authors = nmf_authors.fit_transform(scaled_df)
    nmf_df = pd.DataFrame(data = nmf_authors[:, :2], columns = ['Component 1', 'Component 2'])
    nmf_df = pd.concat([nmf_df, pd.DataFrame(data = dfidx_vals, columns = ['target'])], axis = 1)

    # Plots the transformed data
    plot_data(nmf_df, 'NMF')

def mds(scaled_df, dfidx_vals, distance):
    '''
    Fits an MDS model to the data and plots it

    Inputs:
        scaled_df: the preprocessed DataFrame
        dfidx_vals: corresponding author for each row in the DataFrame
        distance: string representing the distance to use. Examples are:
            'euclidean', 'manhattan', 'chebyshev', and 'canberra'. A full list
            can be found on scikit-learn's documentation for MDS

    Outputs:
        Plots the data after dimension reduction via MDS
    '''
    # Fits the MDS model
    mds_authors = MDS()
    dist = DistanceMetric.get_metric(distance)
    scaled_df = dist.pairwise(scaled_df)
    mds_fit_transform = mds_authors.fit_transform(scaled_df)

    # Plots the scaled data
    fig = px.scatter(None, x=mds_fit_transform[:,0], y=mds_fit_transform[:,1], opacity=1, color=dfidx_vals)
    fig.update_layout(dict(plot_bgcolor = 'white'))
    fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                    showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                    showline=True, linewidth=1, linecolor='black')
    fig.update_layout(title_text=f"MDS {distance.capitalize()} Distance for Book Chapters")
    fig.show()

def tsne(scaled_df, dfidx_vals):
    '''
    Fits tSNE model and plots the first two dimensions

    Inputs:
        scaled_df: the preprocessed DataFrame
        dfidx_vals: corresponding author for each row in the DataFrame

    Outputs:
        Plots the data after dimension reduction via tSNE
    '''
    # Fits the tSNE model
    tsne_authors = TSNE().fit_transform(scaled_df)
    tsne_df = pd.DataFrame(data = tsne_authors[:, 0:2], columns = ['Component 1', 'Component 2'])
    tsne_df = pd.concat([tsne_df, pd.DataFrame(data = dfidx_vals, columns = ['target'])], axis = 1)
    # Plots the data after fitting
    plot_data(tsne_df, 'tSNE')

def spectral_embedding(scaled_df, dfidx_vals):
    '''
    Fits spectral embedding model and plots the first two dimensions

    Inputs:
        scaled_df: the preprocessed DataFrame
        dfidx_vals: corresponding author for each row in the DataFrame

    Outputs:
        Plots the data after dimension reduction via spectral embedding
    '''
    se_authors = SpectralEmbedding().fit_transform(scaled_df)
    se_df = pd.DataFrame(data = se_authors[:, 0:2], columns = ['Component 1', 'Component 2'])
    se_df = pd.concat([se_df, pd.DataFrame(data = dfidx_vals, columns = ['target'])], axis = 1)
    plot_data(se_df, 'Spectral Embedding')


data = pyreadr.read_r('authors.rda')
dfidx_vals, scaled_df = preprocess_data(data, 'authors')
pca_screeplot(scaled_df)
pca_pcs(scaled_df, dfidx_vals)
nmf(scaled_df, dfidx_vals)
mds(scaled_df, dfidx_vals, 'euclidean')
mds(scaled_df, dfidx_vals, 'manhattan')
mds(scaled_df, dfidx_vals, 'chebyshev')
mds(scaled_df, dfidx_vals, 'canberra')
tsne(scaled_df, dfidx_vals)
spectral_embedding(scaled_df, dfidx_vals)
