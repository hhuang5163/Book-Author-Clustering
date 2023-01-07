import pandas as pd
import pyreadr
import numpy as np
from sklearn.decomposition import PCA, NMF
from sklearn import preprocessing
from sklearn.neighbors import DistanceMetric
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
    return dfauthors, scaled_df

def pca(scaled_df, dfauthors):
    '''
    Plots the first two principal components to see if the data
    naturally clusters. Plots the amount of variance explained by each word.

    Inputs: 
        scaled_df: the preprocessed DataFrame
        dfauthors: the original dataset for obtaining labels

    Outputs:
        Plots the data according to the first two principal components and 
        a bar plot representing the amount of variance explained by each word
    '''
    # Fits a PCA model
    pca_authors = PCA(n_components=2).fit_transform(scaled_df)
    pca_df = pd.DataFrame(data = pca_authors, columns = ['Principal Component 1', 'Principal Component 2'])
    pca_df = pd.concat([pca_df, pd.DataFrame(data = dfauthors.index.values, columns = ['target'])], axis = 1)

    # Plots amounts proportional to the variance explained by each word
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Principal Component 1 Over Stop Words')
    ax.set_xlabel('Stop Words')
    ax.set_ylabel('Principal Component 1')
    ax.bar(dfauthors.columns, pca_authors[:, 0])
    plt.xticks(rotation=90)
    fig.savefig('Results Stop Words/WordsPCABar.png')
    plt.show()

    # Plots the words based on the first two principal components
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Principal Component Analysis with 2 PCs')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    targets = dfauthors.columns.values
    ax.scatter(pca_authors[:, 0]
                , pca_authors[:, 1]
                , c = 'b'
                , s = 10)
    for i, txt in enumerate(targets):
        ax.annotate(txt, (pca_authors[i, 0], pca_authors[i, 1]))
    fig.savefig('Results Stop Words/WordsPCAPlot.png')
    plt.show()

def nmf(scaled_df, dfauthors):
    '''
    Plots the first two components to see if the data
    naturally clusters. Plots how important each word is 
    in distinguishing authors

    Inputs: 
        scaled_df: the preprocessed DataFrame
        dfauthors: the original dataset for obtaining labels

    Outputs:
        Plots the data according to the first two components and 
        a bar plot representing how important each word is in 
        distinguishing authors
    '''
    # Scales data so none are negative
    scaled_df = preprocessing.MinMaxScaler().fit_transform(scaled_df)
    # Fits a NMF model
    nmf_authors = NMF()
    nmf_authors = nmf_authors.fit_transform(scaled_df)
    nmf_df = pd.DataFrame(data = nmf_authors[:, :2], columns = ['Component 1', 'Component 2'])
    nmf_df = pd.concat([nmf_df, pd.DataFrame(data = dfauthors.index.values, columns = ['target'])], axis = 1)

    # Plots bar graph to show which words are most important in distinguishing authors
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Stop Words')
    ax.set_ylabel('Component 1')
    ax.set_title('NMF for Stop Words')
    ax.bar(dfauthors.columns, nmf_authors[:, 0])
    plt.xticks(rotation=90)
    fig.savefig('Results Stop Words/WordsNMF1Bar.png')
    plt.show()

    # Plots the stop words according to the first two reduced dimensions
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title('NMF on Stop Words')
    targets = dfauthors.columns.values
    ax.scatter(nmf_authors[:, 0]
                , nmf_authors[:, 1]
                , c = 'b'
                , s = 10)
    for i, txt in enumerate(targets):
        ax.annotate(txt, (nmf_authors[i, 0], nmf_authors[i, 1]))
    fig.savefig('Results Stop Words/WordsNMFPlot.png')
    plt.show()


data = pyreadr.read_r('authors.rda')
dfauthors, scaled_df = preprocess_data(data, 'authors')
scaled_df = scaled_df.transpose()
pca(scaled_df, dfauthors)
nmf(scaled_df, dfauthors)

