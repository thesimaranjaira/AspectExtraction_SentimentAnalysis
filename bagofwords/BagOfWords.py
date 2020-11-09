import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from Utility import Utility
import pandas as pd
import numpy as np
import pickle

if __name__ == '__main__':
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData2.tsv'), header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData2.tsv'), header=0, delimiter="\t", quoting=3)

    raw_input("Press Enter to continue...")

    print 'Download text data sets.'

    clean_train_reviews = []

    print "Cleaning and parsing the training set book reviews...\n"
    for i in xrange(0, len(train["review"])):
        clean_train_reviews.append(" ".join(Utility.review_to_wordlist(train["review"][i], True)))

    print clean_train_reviews

    print "Creating the bag of words...\n"

    # Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)

    # fit_transform() does two functions: First, it fits the model and learns the vocabulary
    # and it transforms our training data into feature vectors.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    filename = 'vec_model2.sav'
    pickle.dump(vectorizer.vocabulary_, open(filename, 'wb'))

    # Numpy arrays are easy to work with, so convert the result to an array
    np.asarray(train_data_features)

    print "Training the random forest (this may take a while)..."
    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 100)

    # Fit the forest to the training set, using the bag of words as features and the sentiment labels
    forest = forest.fit(train_data_features, train["sentiment"])

    filename = 'BOW_model2.sav'
    pickle.dump(forest, open(filename, 'wb'))

    clean_test_reviews = []

    print "Cleaning and parsing the test set book reviews...\n"
    for i in xrange(0,len(test["review"])):
        clean_test_reviews.append(" ".join(Utility.review_to_wordlist(test["review"][i], True)))

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    np.asarray(test_data_features)

    print "Predicting test labels...\n"
    result = forest.predict(test_data_features)

    output = pd.DataFrame(data={"id":test["id"], "sentiment":result})

    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model_new2.csv'), index=False, quoting=3)
    print "Wrote results to Bag_of_Words_model2.csv"
