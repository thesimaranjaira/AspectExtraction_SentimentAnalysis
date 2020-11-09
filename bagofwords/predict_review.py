from sklearn.feature_extraction.text import CountVectorizer
from Utility import Utility
import numpy as np
import pickle

if __name__ == '__main__':

    print "Loading Model (Wait...)"
    # load the model from disk
    filename = 'BOW_model2.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    filename = 'vec_model2.sav'
    voc_loaded = pickle.load(open(filename, 'rb'))

    # Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000, vocabulary=voc_loaded)

    test = raw_input("Enter Review:")

    clean_test_reviews = []

    clean_test_reviews.append(" ".join(Utility.review_to_wordlist(test, True)))

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    np.asarray(test_data_features)

    print "Predicting test label...\n"
    result = loaded_model.predict(test_data_features)
    print result
    for i in result:
        if i==1:
            print "Sentiment predicted: POSITIVE"
        if i==0:
            print "Sentiment predicted: NEGATIVE"
