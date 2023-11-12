#!/usr/bin/env python3

import matplotlib.pyplot as plot
import pickle
import os

if __name__ == "__main__":
    with open(os.sys.argv[1], "rb") as fin:
        data = pickle.load(fin)
    features = {
        "website_len": [],
        "subdomain_len": [],
        "domain_len": [],
        "tld_len": [],
        "website_entropy": [],
        "subdomain_entropy": [],
        "domain_entropy": [],
        "tld_entropy": [],
        "website_metric_entropy": [],
        "subdomain_metric_entropy": [],
        "domain_metric_entropy": [],
        "tld_metric_entropy": [],
        "website_CCR": [],
        "subdomain_CCR": [],
        "domain_CCR": [],
        "tld_CCR": [],
    }

    for v in data.values():
        for k in features.keys():
            features[k].append(v[k])

    for k, v in features.items():
        print("plotting {}".format(k))
        plot.scatter(range(len(v)), v)
        plot.savefig("{}.png".format(k))
        plot.clf()
