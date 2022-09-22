# KMeans_Clustering_Implementation

**K-Means clustering** is the most popular unsupervised machine learning algorithm. K-Means clustering is used to find intrinsic groups within the unlabelled dataset and draw inferences from them. 

So, let's get started.

<a class="anchor" id="0.1"></a>
# **Table of Contents**

1.	[Introduction to K-Means Clustering](#1)
2.  [Applications of clustering](#2)
3.	[K-Means Clustering intuition](#3)
4.	[Choosing the value of K](#4)
5.	[The elbow method](#5)
6.  [Import libraries](#6)
7.	[Import dataset](#7)
8.	[Exploratory data analysis](#8)
9.	[Declare feature vector and target variable](#9)
10.	[Convert categorical variable into integers](#10)
11.	[Feature scaling](#11)
12.	[K-Means model with two clusters](#12)
13.	[K-Means model parameters study](#13)
14.	[Check quality of weak classification by the model](#14)
15.	[Use elbow method to find optimal number of clusters](#15)
16.	[K-Means model with different clusters](#16)
17.	[Results and conclusion](#17)

# **1. Introduction to K-Means Clustering** <a class="anchor" id="1"></a>
[Table of Contents](#0.1)

Machine learning algorithms can be broadly classified into two categories - supervised and unsupervised learning. There are other categories also like semi-supervised learning and reinforcement learning. But, most of the algorithms are classified as supervised or unsupervised learning. The difference between them happens because of presence of target variable. In unsupervised learning, there is no target variable. The dataset only has input variables which describe the data. This is called unsupervised learning.

**K-Means clustering** is the most popular unsupervised learning algorithm. It is used when we have unlabelled data which is data without defined categories or groups. The algorithm follows an easy or simple way to classify a given data set through a certain number of clusters, fixed apriori. K-Means algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity.

K-Means clustering can be represented diagrammatically as follows:-

![image](https://user-images.githubusercontent.com/35486320/191664756-145267ee-2cf6-4f19-9380-0476dfb8e81b.png)

# **2. Applications of clustering** <a class="anchor" id="2"></a>
[Table of Contents](#0.1)
- K-Means clustering is the most common unsupervised machine learning algorithm. It is widely used for many applications which include-

  1. Image segmentation
  2. Customer segmentation
  3. Species clustering
  4. Anomaly detection
  5. Clustering languages

# **3. K-Means Clustering intuition** <a class="anchor" id="3"></a>
[Table of Contents](#0.1)

- K-Means clustering is used to find intrinsic groups within the unlabelled dataset and draw inferences from them. It is based on centroid-based clustering.

- **Centroid** - A centroid is a data point at the centre of a cluster. In centroid-based clustering, clusters are represented by a centroid. It is an iterative algorithm in which the notion of similarity is derived by how close a data point is to the centroid of the cluster.

K-Means clustering works as follows:-
- The K-Means clustering algorithm uses an iterative procedure to deliver a final result. The algorithm requires number of clusters K and the data set as input. 
- The data set is a collection of features for each data point. The algorithm starts with initial estimates for the K centroids. The algorithm then iterates between two steps:-

  ## **3.1 Data assignment step**
  - Each centroid defines one of the clusters. In this step, each data point is assigned to its nearest centroid, which is based on the squared Euclidean distance. So, if ci is the collection of centroids in set C, then each data point is assigned to a cluster based on minimum Euclidean distance. 

  ## **3.2 Centroid update step**
  - In this step, the centroids are recomputed and updated. This is done by taking the mean of all data points assigned to that centroid’s cluster. 
  - The algorithm then iterates between step 1 and step 2 until a stopping criteria is met. Stopping criteria means no data points change the clusters, the sum of the distances is minimized or some maximum number of iterations is reached.
  - This algorithm is guaranteed to converge to a result. The result may be a local optimum meaning that assessing more than one run of the algorithm with randomized starting centroids may give a better outcome.

The K-Means intuition can be represented with the help of following diagram:-

![image](https://user-images.githubusercontent.com/35486320/191665355-9b9db72d-e640-4c71-be38-fd3d78c0a71c.png)

# **4. Choosing the value of K** <a class="anchor" id="4"></a>
[Table of Contents](#0.1)

- The K-Means algorithm depends upon finding the number of clusters and data labels for a pre-defined value of K. To find the number of clusters in the data, we need to run the K-Means clustering algorithm for different values of K and compare the results. 
- So, the performance of K-Means algorithm depends upon the value of K. We should choose the optimal value of K that gives us best performance. There are different techniques available to find the optimal value of K. The most common technique is the **elbow method** which is described below.

# **5. The elbow method** <a class="anchor" id="5"></a>
[Table of Contents](#0.1)

- The elbow method is used to determine the optimal number of clusters in K-means clustering. The elbow method plots the value of the cost function produced by different values of K. The below diagram shows how the elbow method works:-

![image](https://user-images.githubusercontent.com/35486320/191665496-672d7e00-285f-400e-ac67-0ca1c20d4248.png)

We can see that if K increases, average distortion will decrease.  Then each cluster will have fewer constituent instances, and the instances will be closer to their respective centroids. However, the improvements in average distortion will decline as K increases. The value of K at which improvement in distortion declines the most is called the elbow, at which we should stop dividing the data into further clusters.

# **6. Import libraries** <a class="anchor" id="6"></a>
[Table of Contents](#0.1)

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns ....
 
# **7. Import dataset** <a class="anchor" id="7"></a>
[Table of Contents](#0.1)

Dataset Link: https://www.kaggle.com/code/prashant111/k-means-clustering-with-python/data?select=Live.csv

# **8. Exploratory data analysis** <a class="anchor" id="8"></a>
[Table of Contents](#0.1)

- We can see that there are 7050 instances and 16 attributes in the dataset. In the dataset description, it is given that there are 7051 instances and 12 attributes in the dataset.
- So, we can infer that the first instance is the row header and there are 4 extra attributes in the dataset. Next, we should take a look at the dataset to gain more insight about it.

# **9. Declare feature vector and target variable** <a class="anchor" id="9"></a>
[Table of Contents](#0.1)

    X = df
    y = df['status_type']

# **10. Convert categorical variable into integers** <a class="anchor" id="10"></a>
[Table of Contents](#0.1)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X['status_type'] = le.fit_transform(X['status_type'])
    y = le.transform(y)

# **11. Feature Scaling** <a class="anchor" id="11"></a>
[Table of Contents](#0.1)

    from sklearn.preprocessing import MinMaxScaler
    ms = MinMaxScaler()
    X = ms.fit_transform(X)

- **Why Scaling is important?** Scaling of the data makes it easy for a model to learn and understand the problem.

# **12. K-Means model with two clusters** <a class="anchor" id="12"></a>
[Table of Contents](#0.1)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0) 
    kmeans.fit(X)

# **13. K-Means model parameters study** <a class="anchor" id="13"></a>
[Table of Contents](#0.1)
- The KMeans algorithm clusters data by trying to separate samples in n groups of equal variances, minimizing a criterion known as **inertia**, or within-cluster sum-of-squares Inertia, or the within-cluster sum of squares criterion, can be recognized as a measure of how internally coherent clusters are.
- The k-means algorithm divides a set of N samples X into K disjoint clusters C, each described by the mean j of the samples in the cluster. The means are commonly called the cluster **centroids**.
- The K-means algorithm aims to choose centroids that minimize the inertia, or within-cluster sum of squared criterion.

### Inertia
- **Inertia** is not a normalized metric. 
- The lower values of inertia are better and zero is optimal. 
- But in very high-dimensional spaces, euclidean distances tend to become inflated (this is an instance of `curse of dimensionality`). 
- Running a dimensionality reduction algorithm such as PCA prior to k-means clustering can alleviate this problem and speed up the computations.
- The lesser the model inertia, the better the model fit.
- We can see that the model has very high inertia. So, this is not a good model fit to the data.

# **14. Check quality of weak classification by the model** <a class="anchor" id="14"></a>
[Table of Contents](#0.1)

    labels = kmeans.labels_
    correct_labels = sum(y == labels)
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))

# **15. Use elbow method to find optimal number of clusters** <a class="anchor" id="15"></a>
[Table of Contents](#0.1)

    from sklearn.cluster import KMeans
    cs = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(X)
        cs.append(kmeans.inertia_)
    plt.plot(range(1, 11), cs)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('CS')
    plt.show()
    
![image](https://user-images.githubusercontent.com/35486320/191694665-6319a4fd-4d29-4eb1-a72b-1c0802bbc192.png)

- By the above plot, we can see that there is a kink at k=2.
- Hence k=2 can be considered a good number of the cluster to cluster this data.
- But, we have seen that I have achieved a weak classification accuracy of 1% with k=2.
- I will write the required code with k=2 again for convinience.

# **16. K-Means model with different clusters** <a class="anchor" id="16"></a>
[Table of Contents](#0.1)

    kmeans = KMeans(n_clusters=4, random_state=0)
    kmeans.fit(X)
    # check how many of the samples were correctly labeled
    labels = kmeans.labels_
    correct_labels = sum(y == labels)
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
  
# **17. Results and conclusion** <a class="anchor" id="17"></a>
[Table of Contents](#0.1)

1.	In this project, I have implemented the most popular unsupervised clustering technique called **K-Means Clustering**.
2.	I have applied the elbow method and find that k=2 (k is number of clusters) can be considered a good number of cluster to cluster this data.
3.	I have find that the model has very high inertia of 237.7572. So, this is not a good model fit to the data.
4.	I have achieved a weak classification accuracy of 1% with k=2 by our unsupervised model.
5.	So, I have changed the value of k and find relatively higher classification accuracy of 62% with k=4.
6.	Hence, we can conclude that k=4 being the optimal number of clusters.
