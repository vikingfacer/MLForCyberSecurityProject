#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plot
import pandas
import os


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Plot Feature",
        description="Plot the dataset features",
    )
    parser.add_argument("target", help="Pickled Pandas Dataframe of target")
    parser.add_argument("benign", help="Pickled Pandas Dataframe of benign")
    parser.add_argument("-o", help="Output directory for figures", default=None)

    args = parser.parse_args(os.sys.argv[1:])
    with open(args.target, "rb") as fin:
        ads = pandas.read_pickle(fin)

    with open(args.benign, "rb") as fin:
        ben = pandas.read_pickle(fin)

    # redefine ben
    ben = ben.sample(n=int(ads.size / len(ads.columns)))

    features = list(ads.columns)

    for feat in features:
        pandas.DataFrame({"ads": ads[feat], "benign": ben[feat]}).plot.hist(
            alpha=0.5, title=feat
        )
        if args.o:
            plot.savefig("{}/{}.png".format(args.o, feat))
        else:
            plot.savefig("{}.png".format(feat))
        plot.clf()
