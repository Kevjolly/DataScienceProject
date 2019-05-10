import numpy as np
import sys
sys.path.append('../')    # import from folder one directory out
import util
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split


def train_classifier(df, classifier):
    data = df.values.tolist()
    labels = df.index.tolist()
    return classifier.fit(data, labels)


def test_classifier(df, classifier):
    data = df.values.tolist()
    labels = df.index.tolist()
    predictions = classifier.predict(data)
    return balanced_accuracy_score(labels, predictions)*100


def split_df(df):
    msk = np.linspace(0, len(df), len(df)) < len(df)*0.25    # first 25% as val
    val = df[msk]
    train = df[~msk]
    return train, val
    # return train_test_split(df, test_size=0.25)   # for random


def classify_df(train_df, test_df):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf = DecisionTreeClassifier(max_depth=10)
    clf = train_classifier(train_df, clf)
    return test_classifier(test_df, clf)


def main(dataset):
    df = util.csv_to_df(dataset)
    test_df = util.test_df(dataset)
    train_split, val_split = split_df(df)
    clfs = [KNeighborsClassifier(n_neighbors=3),
            DecisionTreeClassifier(max_depth=5)]
    clf = DecisionTreeClassifier(max_depth=10)
    # clf = train_classifier(train_split, clf)
    clf = train_classifier(df, clf)
    # print('Train accuracy: ' + str(test_classifier(train_split, clf)))
    print('Train accuracy: ' + str(test_classifier(df, clf)))
    # print('Val accuracy: ' + str(test_classifier(val_split, clf)))
    print('Test accuracy: ' + str(test_classifier(test_df, clf)))

if __name__ == '__main__':
    if len(sys.argv) == 1:   # dataset not given at command line
        sys.argv.append('../Datasets/star_light/train.csv')
    for dataset in sys.argv[1:]:
        print('')
        print(dataset)
        main(dataset)
