#!/usr/bin/env python3

from tldextract import extract
import pickle
import measures

import argparse
import os
import json
import pathlib


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


def getPickle(pickle_file):
    features = None
    if os.path.isfile(pickle_file):
        with open(pickle_file, "rb") as pickled_obj:
            features = pickle.load(pickled_obj)
    else:
        features = {}
    return features



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Feature Extraction",
        description="extracts features from urls",
    )

    parser.add_argument("DataDirectory", type=pathlib.Path)
    parser.add_argument(
        "--infile", type=argparse.FileType("rb"), help="pickle file imported"
    )
    parser.add_argument(
        "--out", type=argparse.FileType("wb"), help="pickel file output"
    )

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
    if parser.infile:
        extractedFeatures = getPickle(parser.infile)

    for k in features:
        features[k] = {
            **features[k],
            **ExtractFeatures(
                features[k],
                [len, measures.entropy, measures.metric_entropy, measures.CCR],
            ),
        }

    print(features)
    if args.output:
        with open(args.output, "wb") as dataOut:
            pickle.dump(extractedFeatures, dataOut)
