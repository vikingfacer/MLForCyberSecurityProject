#!/usr/bin/env python3

from tldextract import extract
import pickle
import measures

import os
import json


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


def DictListToCSV(d, fn):
    """
    prints keys for header of CSV
    then prints values of dictionary to rows in CSV
    """
    with open(fn, "w") as fout:
        fout.write(",".join([d[iter(d)].keys()]) + "\n")
        for website, v in d.items():
            fout.write(website + "," + ",".join([v[key] for key in v]) + "\n")


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


# next-step-file: filename.split(".")[0] + "Xstep0" + ".cvs", "w")
if __name__ == "__main__":
    data_dir = os.sys.argv[1]

    pickle_file = "{}.pickle".format(data_dir.replace("/", ""))
    features = getPickle(pickle_file)

    for datafile in os.listdir(data_dir):
        print("********** {} ***********".format(datafile))
        with open("{}/{}".format(data_dir, datafile), "r") as fin:
            for site in fin.readlines():
                if NotCommentOrEmpty(site):
                    site = site.replace("\n", "")
                    features[site] = extractAndAdd(site)

    # feature extraction
    for k in features:
        features[k] = {
            **features[k],
            **ExtractFeatures(
                features[k],
                [len, measures.entropy, measures.metric_entropy, measures.CCR],
            ),
        }

    print(features)
    with open(pickle_file, "wb") as dataOut:
        pickle.dump(features, dataOut)
