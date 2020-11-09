import os
import pandas as pd
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    predict = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model_new2.csv'), header=0,  delimiter="\t", quoting=3)
    actual = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'actual_test2.csv'),  header=0,  delimiter="\t", quoting=3)
    act = actual.values.tolist()
    pred = predict.values.tolist()
    print "Accuracy =", accuracy_score(act,pred)*100
