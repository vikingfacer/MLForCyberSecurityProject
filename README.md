# CIS 735

This repo contains the research done for the final paper. The research was to build a classifier
to augment an advertisement blocking DNS sink hole with. The best classifier found through experimenting
was a random forest using lexical analysis on the domain fields of the URL.

## Scripts
There are three main scripts in the repo.
- train model: takes a pickle of benign and target data and trains/test the model the model can be saved as a pickle.
  -- currently train model trains a bunch of models and just scores them
- extract features: consumes the raw data (URLS) as files with -f or directories with -d or just an argument as a positional arg
  -- outputs a pickle for the training model script to consume
- plot features: takes benign and target dataset pickles and plots the features as histograms
  -- plot features outputs a lot of images and -o specifies a directory to put them in

### train_model.py
usage: Train Model [-h] [-b BENIGN] [-t TARGET] [-m MODELNAME]

Uses Benign and target class to train URLs as Target or Benign

options:
  -h, --help            show this help message and exit
  -b BENIGN, --benign BENIGN
                        pickle file of benign
  -t TARGET, --target TARGET
                        pickle file of target class (ads)
  -m MODELNAME, --modelname MODELNAME
                        Output model name

### feature_extraction.py
usage: Feature Extraction [-h] [-i INPUT] [-o OUTPUT] [-c C] [-d] [-f] data

extracts features from urls and output pandas dataframe

positional arguments:
  data

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        pickle file imported
  -o OUTPUT, --output OUTPUT
                        pickle file output
  -c C                  class label to apply on pickle file
  -d                    Directory input flag
  -f                    File input flag

_note: uses measure.py_

### plot_features.py

usage: Plot Feature [-h] [-o O] target benign

Plot the dataset features

positional arguments:
  target      Pickled Pandas Dataframe of target
  benign      Pickled Pandas Dataframe of benign

options:
  -h, --help  show this help message and exit
  -o O        Output directory for figures
