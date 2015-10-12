"""
COMP598 Project 2 Text Analysis

This trains with the training data and produces a predictions csv. Optionally shows validation stuff.

Usage:
  main.py <training-csv> <target-csv> [options]
  main.py compare <csv1> <csv2>

Options:
    --filter=<threshold>    Omit ngrams if they only occur this or fewer number of times [default: 1]
    --ngram-max=<max>       Analyze ngrams up to this length [default: 2]
    --random                The results are always scrambled, but with the same seed. This option makes it a new random each time.
    --validate              Show a percent correct score against the validation data after all processing is done.
    --validation-ratio=<r>  Number from 0 to 1. 0.8 means 80% of data is used for training, 20% for validation. [default: 0.8]

    -h --help              Show this screen.
    -v --version           Show version.
"""

import os.path,time
from docopt import docopt
from constants import *
from utilities import *
from gimli import *

def get_csv_path():
    #scans all existing data csvs, returns the name with the lowest number suffix that is unused
    folder="results"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    index=1
    def get_path(index,folder):
        filename="results-%s.csv"%str(index).zfill(3)
        return os.path.join(folder,filename)
    while os.path.isfile(get_path(index,folder)):
        index+=1
    return get_path(index,folder)

def is_file(path):
    if not os.path.isfile(path):
        print_color("Not a file: '%s'"%path,color=COLORS.RED)
        return 0
    return 1

def compare(csv1,csv2):
    if not is_file(csv1):
        return
    if not is_file(csv2):
        return

    print("Comparing:\n%s\n%s"%(csv1,csv2))
    with open(csv1,"r") as f:
        lines1=f.readlines()
    with open(csv2,"r") as f:
        lines2=f.readlines()

    matches=0
    for i,line1 in enumerate(lines1):
        line2=lines2[i]
        if line1==line2:
            matches+=1
    percent=round(100*matches/len(lines1),3)
    print("%s%% of lines match."%percent)

def main(args):
    random.seed(123)

    if args["compare"]:
        compare(args["<csv1>"],args["<csv2>"])
        return

    training=args["<training-csv>"]
    if not is_file(training):
        return
    target=args["<target-csv>"]
    if not is_file(target):
        return

    try:
        threshold=int(args["--filter"])
    except ValueError:
        print_color("Bad value for filter.")
        return

    try:
        validation_ratio=float(args["--validation-ratio"])
    except ValueError:
        print_color("Bad value for validation ratio.")
        return

    try:
        ngram_max=int(args["--ngram-max"])
    except ValueError:
        print_color("Bad value for ngram max.")
        return

    gimli=Gimli(training,filter_threshold=threshold,
            ngram_max=ngram_max,
            validate=args["--validate"],
            validation_ratio=validation_ratio)
    print_color("Making predictions.",COLORS.GREEN)
    gimli.make_predictions_csv(target)

    print("\nDone.\n")

if __name__ == "__main__":
    args = docopt(__doc__, version="1.0")
    main(args)
















