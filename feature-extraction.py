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


def notCommentOrEmpty(line):
    """
    helper function to sanitize lines
    returns bool if line is not comment or empty
    """
    return line[0] != "#" and len(line) > 0 and line[0] != "\n"


def extractFeatures(websiteDict, fn):
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


def extractDatFromFile(file):
    """
    reads in file removes line ending and seperates the Url into fields
    w/extractAndAdd
    returns list of 'extractAndAdd dictionaries'
    """
    cleanizedData = []
    with open(file, "r") as fin:
        for site in fin.readlines():
            if notCommentOrEmpty(site):
                site = site.replace("\n", "")
                cleanizedData.append(extractAndAdd(site))
    return cleanizedData


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


def applyMeasurements(featureMap, cleanedRawData, extractedFeatures):
    for k in featureMap:
        for fn in featureMap[k]:
            extractedFeatures["{}_{}".format(k, fn.__name__)] = [
                fn(x[k]) for x in cleanedRawData
            ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Feature Extraction",
        description="extracts features from urls and output pandas dataframe",
    )

    parser.add_argument(
        "data",
        type=str,
    )
    parser.add_argument(
        "-i", "--input", type=pathlib.Path, help="pickle file imported", default=None
    )
    parser.add_argument(
        "-o", "--output", type=pathlib.Path, default=None, help="pickle file output"
    )
    parser.add_argument("-c", type=int, help="class label to apply on pickle file")
    parser.add_argument("-d", action="store_true", help="Directory input flag")
    parser.add_argument("-f", action="store_true", help="File input flag")

    args = parser.parse_args(os.sys.argv[1:])

    cleanedRawData = []
    if args.d:
        for datafile in os.listdir(args.data):
            print("********** {} ***********".format(datafile))
            dataFileStr = "{}/{}".format(args.DataDirectory, datafile)
            cleanedRawData += extractDatFromFile(dataFileStr)
    elif args.f:
        # just open file and extract to pickle
        cleanedRawData += extractDatFromFile(args.data)
    else:
        cleanedRawData += [extractAndAdd(args.data)]

    extractedFeatures = {}
    if args.input:
        extractedFeatures = pandas.read_pickle(args.input).to_dict()

    applyMeasurements(features, cleanedRawData, extractedFeatures)

    # add label for training classification
    if args.c:
        # add the class label
        extractedFeatures["classification"] = [args.c] * len(
            list(extractedFeatures.values())[0]
        )

    # put into a pandas dataframe

    if args.output:
        with open(args.output, "wb") as dataOut:
            pickle.dump(pandas.DataFrame(extractedFeatures), dataOut)
    else:
        print(extractedFeatures)
