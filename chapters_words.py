import pandas as pd
import pyreadr
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import SpectralBiclustering
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

def biclustering(scaled_df, dfauthors):
    '''
    Biclustering for both the book chapters and stop words simultaneously

    Inputs: 
        scaled_df: the preprocessed DataFrame
        dfauthors: the original dataset for obtaining labels

    Outputs:
        Plots the data after biclustering and after rearranging to
        obtain a checkerboard structure
    '''
    # Fits the Biclustering model
    n_clusters = (4, 4)
    model = SpectralBiclustering(n_clusters=n_clusters)
    model.fit(scaled_df)
    scaled_df = scaled_df.to_numpy()
    fit_data = scaled_df[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]

    # Plots the biclustering results
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(fit_data, cmap=plt.cm.Blues, aspect='auto')
    ax.set_title("Biclustering on Book Chapters and Stop Words")
    ax.set_xticks(np.arange(dfauthors.columns.size))
    ax.set_yticks(np.arange(0, dfauthors.index.values.size, 30))
    ax.set_xticklabels(dfauthors.columns[np.argsort(model.column_labels_)], rotation=90)
    ax.set_yticklabels(dfauthors.index.values[np.argsort(model.row_labels_)][::30])
    fig.savefig('Results Chapters & Words/BothBookWordsBiClustering1.png')
    plt.show()

    # Rearranges the biclustered data to obtain a checkerboard structure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(
        np.outer(np.sort(model.row_labels_) + 1, np.sort(model.column_labels_) + 1),
        cmap=plt.cm.Blues,
        aspect='auto'
    )
    ax.set_title("Checkerboard structure of rearranged data")
    ax.set_xticks(np.arange(dfauthors.columns.size))
    ax.set_yticks(np.arange(0, dfauthors.index.values.size, 30))
    ax.set_xticklabels(dfauthors.columns[np.argsort(model.column_labels_)], rotation=90)
    ax.set_yticklabels(dfauthors.index.values[np.argsort(model.row_labels_)][::30])
    fig.savefig('Results Chapters & Words/BothBookWordsBiClustering2.png')
    plt.show()


data = pyreadr.read_r('authors.rda')
dfauthors, scaled_df = preprocess_data(data, 'authors')
biclustering(scaled_df, dfauthors)
