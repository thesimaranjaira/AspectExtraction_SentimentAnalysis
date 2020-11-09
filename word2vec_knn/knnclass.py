import pickle
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import dataclean as dc
import classifierFuncs as cfun
import time


def clean_data(data):
    # raise an error if there is no review column
    try:
        reviewsSet = data["review"]
    except ValueError:
        print('No "review" column!')
        raise

    cleaned_data = [dc.review_to_words(review, True, True, False) for review in reviewsSet]

    return cleaned_data


def main():

    model = Word2Vec.load('classifier/Word2VectforNLPTraining2')

    wordVectors = model.syn0

    num_clusters = int(wordVectors.shape[0] / 5)

    '''
    print("First Time Clustering...")
    startTime = time.time()
    clusterIndex = cfun.kmeans(num_clusters, wordVectors)
    endTime = time.time()

    print("Time taken for clustering: {} seconds".format(endTime - startTime))

    clusterf = open("classifier/word2vec/clusterIndex2.pickle","wb")
    
    pickle.dump(clusterIndex, clusterf)

    '''
    filename = 'classifier/word2vec/clusterIndex2.pickle'
    clusterIndex = pickle.load(open(filename, 'rb'))

    # create a word/index dictionary, mapping each vocabulary word to a cluster number
    # zip(): make an iterator that aggregates elements from each of the iterables
    index_word_map = dict(zip(model.index2word, clusterIndex))

    train = pd.read_csv("data/labeledTrainData2.tsv",
                    header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("data/testData2.tsv",
                   header=0, delimiter="\t", quoting=3)

    trainingDataFV = np.zeros((train["review"].size, num_clusters), dtype=np.float)
    testDataFV = np.zeros((test["review"].size, num_clusters), dtype=np.float)

    # We dont really need to clean the data as the junk terms will be ignored anyway. This is due to the fact that we did not consider these while creating the model
    # and hence they will not feature in the model's vocabulary. Still this step will expedite the classification and feature vector creation.
    print("Processing training data...")
    counter = 0
    cleanedTrainingData = clean_data(train)
    for review in cleanedTrainingData:
        trainingDataFV[counter] = cfun.create_bag_of_centroids(review, num_clusters, index_word_map)
        counter += 1

    print("Processing test data...")
    counter = 0
    cleaned_test_data = clean_data(test)
    for review in cleaned_test_data:
        testDataFV[counter] = cfun.create_bag_of_centroids(review, num_clusters, index_word_map)
        counter += 1

    print("Predicting Sentiments...")
    n_neighbour = 5
    result = cfun.knnClassifer(n_neighbour, trainingDataFV, train["sentiment"], testDataFV)

    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("Word2Vec_knn_new2.csv", index=False, quoting=3)
    print "Wrote results to Word2Vec_knn_new2.csv"


if __name__ == '__main__':
    main()