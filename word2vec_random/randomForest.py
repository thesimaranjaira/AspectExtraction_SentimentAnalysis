import pickle
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import dataclean as dc
import time
import classifierFuncs as cfun


def clean_data(data):
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
    # print(wordVectors[0])
    num_clusters = int(wordVectors.shape[0] / 5)

    print("Clustering...")

    '''
    print("Clustering First Time...")
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
    n_estimators = 100
    result = cfun.rfClassifer(n_estimators, trainingDataFV, train["sentiment"], testDataFV)
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("Word2Vec_random_new2.csv", index=False, quoting=3)
    print "Wrote results to Word2Vec_random_new2.csv"

if __name__ == '__main__':
    main()