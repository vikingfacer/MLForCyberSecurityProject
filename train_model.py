#!/usr/bin/env python3

import argparse
import os
import pathlib

import pickle
import numpy as np
import pandas
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

"""
benign set is 1 Million rows compared to target set is 154,000
target size will be used to sample benign set
"""


def makeDataSet(benign, target):
    # benign class is a giant set
    # there for only a portion equal to the size of target is selected
    benign_sample = benign.sample(n=int(target.size / len(target.columns)))

    # combine the two sets
    model_data = pandas.concat([target, benign_sample])

    # Step to move classification to Y from X
    Y = model_data.classification
    X = model_data.drop("classification", axis=1)

    return X, Y


def plotFeatImportance(randomForest, labels, ax):
    """
    plots the importances of features determined by the Random Forest
    randomForest: randomforest w/ feature importance
    labels: labels of features
    ax: plot to graph feature importance on
    """
    std = np.std(
        [tree.feature_importances_ for tree in randomForest.estimators_], axis=0
    )
    importance = pandas.Series(std, labels)
    ax.set_title("Random Forest Feature importances using MDI")
    importance.plot.bar(yerr=std, ax=ax)


def plotConfusionMatrix(randomForest, testX, testY, classlabels):
    ConfusionMatrixDisplay.from_estimator(
        randomForest, testX, testY, display_labels=classlabels
    )


if "__main__" == __name__:
    parser = argparse.ArgumentParser(
        prog="Train Model",
        description="Uses Benign and target class to train URLs as Target or Benign",
    )

    parser.add_argument(
        "-b", "--benign", type=pathlib.Path, help="pickle file of benign"
    )
    parser.add_argument(
        "-t", "--target", type=pathlib.Path, help="pickle file of target class (ads)"
    )
    parser.add_argument("-m", "--modelname", type=str, help="Output model name")

    args = parser.parse_args(os.sys.argv[1:])

    with open(args.target, "rb") as fin:
        target_class = pandas.read_pickle(fin)

    with open(args.benign, "rb") as fin:
        benign_class = pandas.read_pickle(fin)

    X, Y = makeDataSet(benign_class, target_class)

    classifier = {
        "randomforest": RandomForestClassifier(10),
        "decsiontree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "knearestnieghbor": KNeighborsClassifier(3),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "NeuralNet": MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    }
    for split in [0.3, 0.4, 0.5, 0.6, 0.7]:
        print("Test split {}".format(split))
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=split)
        for name, c in classifier.items():
            print("training {}".format(name))
            c.fit(train_x, train_y)
            print("testing {}".format(name))
            scores = c.score(test_x, test_y)
            print(scores)

            predictions = c.predict(test_x)
            print(
                classification_report(
                    test_y, predictions, target_names=["benign", "ads"]
                )
            )

            pngfn = "{}0-split-{}-{}.png"
            classlabels = ["ads", "benign"]
            plotConfusionMatrix(c, test_x, test_y, classlabels)
            plt.savefig(pngfn.format(int(split), name, "ConfusionMatrix"))
            plt.clf()
