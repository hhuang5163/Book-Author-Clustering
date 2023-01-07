# Book-Author-Clustering
Data obtained from https://github.com/DataSlingers/clustRviz/tree/master/data. The data file is named <b>authors.rda</b>.

## Data Analysis
### Clustering by Book Chapters
<img src="https://github.com/hhuang5163/Book-Author-Clustering/blob/main/Results%20Book%20Chapters/PCA.png">
<img src="https://github.com/hhuang5163/Book-Author-Clustering/blob/main/Results%20Book%20Chapters/tSNE.png">
<br>
The best visual summary of the data for distinguishing authors based on book chapters is PCA since it reasonably separates the four authors into four clusters. This visual would be greatly aided with the tSNE visualization for book chapters since it provides a better visual of the four clusters separated out.<br>
Combined, the PCA reinforces the relationships found by the tSNE clustering which focuses on minimizing distances between points and their closest neighbors for visualization. Of course, since this is a clustering task to distinguish the four authors, we can see that tSNE has accomplished this pretty well. However, since tSNE loses any notion of distance, the PCA visualization can be used to see that while the book chapters are similar to each other in terms of the frequency of stop words, there is still a fairly clear divide between the authors (with the exception of Shakespeare and Milton). Even if all the points representing book chapters were colored the same color (the visual colors them based on the correct author), we can still see that Austen and London are clearly clustered.<br>
<img src="https://github.com/hhuang5163/Book-Author-Clustering/blob/main/Results%20Book%20Chapters/Spectral%20Embedding.png">
<br>
The other methods are less informative as the divide between different authors is less emphasized; for example, spectral embedding puts Milton and Shakespeare close together. Compared with tSNE, it’s evident that tSNE does better at separating these authors.<br>
<img src="https://github.com/hhuang5163/Book-Author-Clustering/blob/main/Results%20Book%20Chapters/MDSCanberra.png">
<img src="https://github.com/hhuang5163/Book-Author-Clustering/blob/main/Results%20Book%20Chapters/MDSManhattan.png">
<img src="https://github.com/hhuang5163/Book-Author-Clustering/blob/main/Results%20Book%20Chapters/MDSEuclidean.png">
<img src="https://github.com/hhuang5163/Book-Author-Clustering/blob/main/Results%20Book%20Chapters/MDSChebyshev.png">
<br>
MDS Canberra distance works fairly well but yields approximately the same results as PCA, with Shakespeare and Milton clustered close together. MDS with Manhattan distance also works well in clustering into 4 clusters but distinguishing between Austen and London is harder. From the other visualizations, we can clearly see that MDS Euclidean and MDS Chebyshev do not work as well.<br>
<img src="https://github.com/hhuang5163/Book-Author-Clustering/blob/main/Results%20Book%20Chapters/NMF.png">
<br>
Lastly, NMF was not chosen for the same reason as the other MDS distances: the clusters are not clearly distinguishable to show a difference between the four authors.
These patterns are enough to distinguish the four authors. There are certainly outliers, as seen in both the PCA and tSNE visualizations. We have two London book chapters that are “closer” to Austen’s book chapters and one of London’s book chapters that is closer to Shakespeare. This can also be seen in the PCA plot, although it is not as clearly emphasized as in the tSNE visualization.<br>
Jane Austen (1775-1817) and Jack London (1876-1916) were close to each other in terms of when they wrote their books, so it makes sense that their language would be similar, which would explain how some of London’s books were clustered with Austen’s books. This can also be seen in the PCA visualization, where London and Austen were clustered close to each other.

### Clustering by Stop Words
<img src="https://github.com/hhuang5163/Book-Author-Clustering/blob/main/Results%20Stop%20Words/WordsPCA1Bar.png">
<img src="https://github.com/hhuang5163/Book-Author-Clustering/blob/main/Results%20Stop%20Words/WordsPCA2Bar.png">
<br>
Notice that here, certain words are given higher pattern/principal component values than others. This implies that certain stop words, such as “the” explain more variance in PC1 between the authors than other words such as “her.” Combined with the graphs from the book chapters part, we can reveal some interesting relationships amongst the stop words.
<img src="https://github.com/hhuang5163/Book-Author-Clustering/blob/main/Results%20Stop%20Words/WordsPCAPlot.png">
<br>
The best visual summary of the data for this task is the PCA graphs. This is because the PCA bar graph illustrates the amount of variance explained by each word for each principal component. It also keeps words closer to each other relatively close. However, the PCA plot was not fully conducive to understanding these words, so NMF was also used on these stop words to exaggerate neighboring relationships. Interestingly, we see that words that are commonly used together are grouped together; for example, “would” and “have” are grouped very closely. Unsurprisingly, we see that a lot of prepositional words such as “on”, “by”, “but”, etc. are grouped together near 0 for both component 1 and 2, which we can actually interpret as meaning all books used these words approximately in the same manner; hence using the book chapters as features clustered them close together. We can also see a lot of the variance is explained just by the first component for both PCA and NMF.<br>
Using the PCA bar plots for the stop words, we can draw interesting connections between the authors and the books. For example, looking at “her”, this word has a negative value for PC1, which correlates with the fact that Austen was more on the negative side of PC1 from the first part. She wrote more books centered around a female main character and so it stands to reason that she would use “her” particularly often. The same can be said of the word “your.” Austen and Shakespeare both wrote pieces concerning people, and as such, often used “your,” which corresponds to the negative PC1 value for “your.” On the other hand, Jack London wrote books about realism and naturalism, both topics in which the word “your” probably would not appear too often.<br>
A note on the uniqueness of “the” as well: In both PCA and NMF, “the” is fairly distant from other words yet has a very high component 1 value: this aligns with the earlier notion that words of the same type are grouped together and words that are used together are grouped together; the word “the” usually is followed by a noun, and most of these words are not nouns and thus should not be clustered with “the.” For example, the phrase “the he” is not commonly seen in literature if at all.<br>

### Biclustering
<img src="https://github.com/hhuang5163/Book-Author-Clustering/blob/main/Results%20Chapters%20%26%20Words/BothBookWordsBiClustering1.png">
<br>
As we can see from the biclustering visualization (darker blue squares imply higher counts of words in that book chapter), we have a slight checkerboard pattern that clearly distinguishes the words that certain authors use more than other. These imply that that author used that word often. For example, for Jane Austen, we see this darker rectangle for all her book chapters from the word “has” to “been.” This means that Austen used these words more than the other authors since this region of the visualization is much darker than the areas surrounding it. <br>
The stop words that are helpful for distinguishing _____’s book chapters are the words ranging from:
<ul>
<li>Austen: “has” to “been”
<li>Milton: “their” to “with”
<li>Shakespeare: “so” to “our”
<li>London: “one” to “a”
</ul>
In this biclustering diagram, we can see that we have a checkerboard layout. This implies that these words are fairly “clean-cut” between the authors, meaning each author either uses a high concentration of these words compared to the others, or they do not. We notice that London also shares a high usage of the words from “their” to “with” with Milton since that area is also darker on the diagram. However, in terms of identifying Milton’s works from London’s works, we can easily see that Milton does not use the words “one” to “a” whereas London very frequently does.<br>

### Clustering Accuracy
<img src="https://github.com/hhuang5163/Book-Author-Clustering/blob/main/Clustering%20Accuracy/ClusteringKMeans4.png"><br>
K-Means performs fairly well, but clusters incorrectly some book chapters. However, K-Means performs best with Gaussian distributions with compact, balanced clusters. These clusters and not compact and certainly not balanced and as a result, it is reasonable that K-Means does not perform as well.<br>
<img src="https://github.com/hhuang5163/Book-Author-Clustering/blob/main/Clustering%20Accuracy/HC4wardeuclidean.png"><br>
It is reasonable that hierarchical clustering with linkage ward and Euclidean distance performs worse on this data since it produces very balanced clusters. Given that we have much more of Austen and London’s works than Milton’s or Shakespeare’s, its reasonable that this clustering method performs worse on this authors data.<br>
### Cluster Validation
<img src="https://github.com/hhuang5163/Book-Author-Clustering/blob/main/Silhouette%20Plots/HCwardeuclidean4.png"><br>
Hierarchical Clustering, Linkage: Ward, Distance: Euclidean<br>
For 2 clusters, the average silhouette score is: 0.11179590416440538<br>
For 3 clusters, the average silhouette score is: 0.13092258606841448<br>
For 4 clusters, the average silhouette score is: 0.13833900164118343<br>
For 5 clusters, the average silhouette score is: 0.12063475116542396<br>
For 6 clusters, the average silhouette score is: 0.09618812489451299<br>
<img src="https://github.com/hhuang5163/Book-Author-Clustering/blob/main/Silhouette%20Plots/KMeans4.png"><br>
K-Means Clustering<br>
For 2 clusters, the average silhouette score is: 0.11692099014923246<br>
For 3 clusters, the average silhouette score is: 0.1316417958433564<br>
For 4 clusters, the average silhouette score is: 0.14068119186674763<br>
For 5 clusters, the average silhouette score is: 0.12435743480600557<br>
For 6 clusters, the average silhouette score is: 0.10471537948038324<br>

From these statistics, we see that hierarchical clustering using linkage ward and Euclidean distance selected K = 4 and K-Means clustering also selected K = 4.<br>
The silhouette plot indicates the size of each cluster and the silhouette coefficient for each observation in each cluster. The blue line indicates the average silhouette coefficient for the entire clustering with K clusters.<br>
Based off of the silhouette plot, we can see that the silhouette plot for any cluster that was not selected had most of its clusters below average, and one cluster that is typically much more well classified. For the cluster K that was selected, we can see that more of the silhouette scores per cluster are past the average coefficient line, hence it is a better fit for the model.<br>
Both K-Means and hierarchical clustering with ward linkage and Euclidean distance perform as expected since they selected K = 4; these methods are both suited for balanced and compact clusters and as a result, they are good for selecting the number of clusters (although in doing so will mis-classify some authors). However, in terms of accuracy, they would not be as good for classifying points to clusters as noted previously; for this authors dataset, there are methods better suited for classification than K-Means or hierarchical clustering.
## To run
To run the code, open the file and run it to replicate the results. Sample results for clustering can be found inside the corresponding results folders.<br>
<ul>
<li><b>book_chapters.py</b> contains code that will cluster the authors dataset by book chapters
<li><b>chapters_words.py</b> contains code to cluster the authors dataset using stop words
<li><b>clustering_accuracy.py</b> will determine the accuracy of clustering with different models
<li><b>cluster_validation.py</b> will perform hyperparameter tuning to determine which models will correctly tune the number of clusters in the data to 4.
</ul>
