"""
COMP598 Project 2 Text Analysis

This trains with the training data and produces a predictions csv. Optionally shows validation stuff.

Usage:
  main.py gimli <training-csv> [<target-csv>] [options]
  main.py pippin <training-csv> [<target-csv>] [options]
  main.py frodo <training-csv> [<target-csv>] [options]

Options:
    --ngram-max=<max>       Analyze ngrams up to this length [default: 2]
    --random                The results are always scrambled, but with the same seed. This option makes it a new random each time.
    --validate              Show a percent correct score against the validation data after all processing is done.
    --validation-ratio=<r>  Number from 0 to 1. 0.8 means 80% of data is used for training, 20% for validation. [default: 0.8]
    --timer=<interval>      Wait this number of seconds before showing an update on processing. [default: 20]
    --balanced              Cull training data so that there is an even number of each category.

    --filter=<threshold>    (Gimli) Omit ngrams if they only occur this or fewer number of times [default: 1]
    --k=<k>                 (Pippin) Count this many nearest neighbours. [default: 5]

    -h --help              Show this screen.
    -v --version           Show version.
"""

import os.path,time,random
from docopt import docopt
from constants import *
from utilities import *
from gimli import *
from pippin import *
from SimpleNB_fordo import NB as Frodo

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

def main(args):
    if not args["--random"]:
        random.seed(123)

    training=args["<training-csv>"]
    if not is_file(training):
        return

    target=args["<target-csv>"]
    if target and not is_file(target):
        return

    try:
        threshold=int(args["--filter"])
    except ValueError:
        print_color("Bad value for filter.")
        return

    try:
        k=int(args["--k"])
    except ValueError:
        print_color("Bad value for k.")
        return

    try:
        interval=int(args["--timer"])
    except ValueError:
        print_color("Bad value for timer interval.")
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

    if args["gimli"]:
        ta=Gimli(training,validation_ratio=validation_ratio,interval=interval,balanced=args["--balanced"])
        ta.process(ngram_max=ngram_max,filter_threshold=threshold)
    if args["pippin"]:
        ta=Pippin(training,validation_ratio=validation_ratio,interval=interval,balanced=args["--balanced"])
        ta.process(ngram_max=ngram_max,k=k)
    if args["frodo"]:
        ta=Frodo(training,validation_ratio=validation_ratio,interval=interval,balanced=args["--balanced"])
        ta.process()
        
    if args["--validate"]:
        print_color("Starting validation.",COLORS.GREEN)
        ta.validate()
    if target:
        print_color("Making predictions.",COLORS.GREEN)
        ta.make_predictions_csv(target)

    print("\nDone.\n")

if __name__ == "__main__":
    args = docopt(__doc__, version="1.0")
    main(args)
















