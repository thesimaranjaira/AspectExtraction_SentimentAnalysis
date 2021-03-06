import logging
from sklearn.cluster import KMeans
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def kmeans(num_clusters, dataSet):
    # n_clusters: number of centroids
    # n_jobs: number of jobs running in parallel
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
   
    kmeansClustering = KMeans(n_clusters=num_clusters)
    # Compute cluster centers and predict cluster index for each sample
    centroidIndx = kmeansClustering.fit_predict(dataSet)

    return centroidIndx


def create_bag_of_centroids(reviewData,num_clusters,index_word_map):
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    """
        assign each word in the review to a centroid
        this returns a numpy array with the dimension as num_clusters
        each will be served as one feature for classification
        :param reviewData:
        :return:
    """
    featureVector = np.zeros(num_clusters, dtype=np.float)
    
    for word in reviewData:
        if word in index_word_map:
            index = index_word_map[word]
            featureVector[index] += 1
    
    return featureVector


def knnClassifer(n_neighbors, trainingSet, label, testSet):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    clf = KNeighborsClassifier(n_neighbors, n_jobs=-1)
    clf = clf.fit(trainingSet, label)
    print "Predicting now.."
    result = clf.predict(testSet)

    return result
