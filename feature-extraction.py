#!/usr/bin/env python3

from tldextract import extract
import pickle
import measures

import argparse
import os
import json
import pathlib
import pandas


def extractAndAdd(website):
    """
    returns dict{
    "website": website
    "subdomain": subdmain if one exists
    "domain": domain
    "tld": top level domain
    }
    """
    Keys = ["website", "subdomain", "domain", "tld"]
    x = extract(website)
    # suffix = tld
    return {
        Keys[0]: website,
        Keys[1]: x.subdomain,
        Keys[2]: x.domain,
        Keys[3]: x.suffix,
    }


def NotCommentOrEmpty(line):
    """
    helper function to sanitize lines
    returns bool if line is not comment or empty
    """
    return site[0] != "#" and len(site) > 0 and site[0] != "\n"


def ExtractFeatures(websiteDict, fn):
    """
    creates additional dictionary of {key + func name : func applied}
    this is essentially a list map function for dictionaries
    returns the dictionary
    """
    extracted_features = {}
    for func in fn:
        for k in websiteDict.keys():
            extracted_features["{}_{}".format(k, func.__name__)] = func(websiteDict[k])
    return extracted_features


def apply(datalist, part, fn):
    return {"{}_{}".format(part, fn): [fn(d[part]) for d in datalist]}


def getPickle(pickle_file):
    features = None
    if os.path.isfile(pickle_file):
        with open(pickle_file, "rb") as pickled_obj:
            features = pickle.load(pickled_obj).to_dict()
    else:
        features = {}
    return features


"""
    features: extracted from raw data
    key: data field
    value: list of measures
    key+measure = feature
"""
features = {
    "website": [len, measures.entropy, measures.metric_entropy, measures.CCR],
    "subdomain": [len, measures.entropy, measures.metric_entropy, measures.CCR],
    "domain": [len, measures.entropy, measures.metric_entropy, measures.CCR],
    "tld": [len, measures.entropy],
}


parser = argparse.ArgumentParser(
    prog="Feature Extraction",
    description="extracts features from urls and output pandas dataframe",
)

parser.add_argument(
    "DataDirectory",
    type=pathlib.Path,
)
parser.add_argument("-i", type=pathlib.Path, help="pickle file imported")
parser.add_argument("-o", type=pathlib.Path, help="pickle file output")
parser.add_argument("-c", type=int, help="class label to apply on pickle file")

if __name__ == "__main__":
    args = parser.parse_args(os.sys.argv[1:])

    cleanedRawData = []
    for datafile in os.listdir(args.DataDirectory):
        print("********** {} ***********".format(datafile))
        with open("{}/{}".format(args.DataDirectory, datafile), "r") as fin:
            for site in fin.readlines():
                if NotCommentOrEmpty(site):
                    site = site.replace("\n", "")
                    cleanedRawData.append(extractAndAdd(site))

    extractedFeatures = {}
    if args.i:
        extractedFeatures = getPickle(args.i)

    for k in features:
        for fn in features[k]:
            extractedFeatures["{}_{}".format(k, fn.__name__)] = [
                fn(x[k]) for x in cleanedRawData
            ]
    # add the class label
    extractedFeatures["classification"] = [args.c] * len(
        list(extractedFeatures.values())[0]
    )

    # put into a pandas dataframe

    if args.o:
        with open(args.o, "wb") as dataOut:
            pickle.dump(pandas.DataFrame(extractedFeatures), dataOut)
