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
from sklearn.ensemble import RandomForestClassifier


"""
benign set is 1 Million rows compared to target set is 154,000
target size will be used to sample benign set
"""


def makeDataSet(benign, target):
    # benign class is a giant set
    # there for only a portion equal to the size of target is selected
    benign_sample = benign_class.sample(
        n=int(target_class.size / len(target_class.columns))
    )

    # combine the two sets
    model_data = pandas.concat([target_class, benign_sample])

    # Step to move classification to Y from X
    Y = model_data.classification
    X = model_data.drop("classification", axis=1)

    return X, Y


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

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3)

    rfc = RandomForestClassifier(10)
    print("training {}".format(type(rfc)))
    rfc.fit(train_x, train_y)
    print("testing {}".format(type(rfc)))
    scores = rfc.score(test_x, test_y)
    print("score of {}".format(type(rfc)))
    print(scores)

    if args.modelname:
        print("Saving Model: as {}".format(args.modelname))
        with open(args.modelname, "wb") as fout:
            pickle.dump(rfc, fout)
