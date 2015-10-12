"""
COMP598 CSV Cleaner

Usage:
    cleaner.py <csv> [options]

Options:
    -h --help              Show this screen.
    -v --version           Show version.
"""

import os.path, re, csv
from docopt import docopt
from nltk.stem.snowball import SnowballStemmer
from constants import *
from utilities import *

def is_file(path):
    if not os.path.isfile(path):
        print_color("Not a file: '%s'"%path,color=COLORS.RED)
        return 0
    return 1

def tokenize_text(text):
    text=text.replace("__EOS__","_eos")
    text=delete_wacky_chars(text)
    text=re.sub("[.]{2,}"," _ellipses ",text)
    text=re.sub("[0-9]+"," # ",text)
    for c in TOKENS:
        token=TOKENS[c]
        text=text.replace(c,token)

    return re.split("[ ]+",text.strip())    

def clean_text(text,stemmer=0,
        remove_stops=True,
        remove_tokens=True,
        remove_periods=True):
    #default behaviour is most aggressive strip and clean
    words=tokenize_text(text)
    new_words=[]
    previous_word="_period"
    for word in words[:]:
        if not word:
            continue
        keep_caps=is_caps_meaningful(previous_word,word)
        if not keep_caps:
            word=word.lower()
        previous_word=word

        if remove_stops and word in ENGLISH_STOPWORDS:
            continue
        if remove_tokens and word.startswith("_") and word != "_period":
            continue
        if remove_periods and word=="_period":
            continue

        if not keep_caps and stemmer and not word.startswith("_"):
            word=stemmer.stem(word)
        new_words.append(word)

    return " ".join(new_words)

def clean_csv(args,source_csv):
    ss=SnowballStemmer('english')
    data=get_data(source_csv)

    folder="clean"+os.sep
    if not os.path.isdir(folder):
        os.mkdir(folder)

    columns,header=get_file_info(source_csv)

    for i in range(16):
        b=[int(i) for i in bin(i)[2:].zfill(4)]
        remove_stops=b[0]
        use_stemmer=b[1]
        remove_tokens=b[2]
        remove_periods=b[3]

        filename=source_csv.split(os.sep)[-1].split(".csv")[0]
        filename+="-%s"%"".join([str(i) for i in b])+".csv"
        print_color("Cleaning '%s'"%filename,COLORS.YELLOW)
        new_data=[]
        for item in data:
            id,text=item[0],item[1]
            if columns==3:
                category=item[2]
            text=clean_text(text,stemmer=ss if use_stemmer else 0,
                        remove_stops=remove_stops,
                        remove_tokens=remove_tokens,
                        remove_periods=remove_periods)
            if columns==3:
                new_data.append((id,text,category))
            else:
                new_data.append((id,text))

        with open(folder+filename,"w") as f:
            f.write(header)
            for item in new_data:
                stringy=[str(i) for i in item]
                f.write("\n"+",".join(stringy))

def get_file_info(source_csv):
    if "train" in source_csv:
        columns=3
        header="Id,Interview,Prediction"
    else:
        columns=2
        header="Id,Interview"
    return columns,header

def get_data(source_csv):
    data=[]
    columns,header=get_file_info(source_csv)

    with open(source_csv,newline="") as f:
        reader=csv.reader(f)
        is_header=True
        for row in reader:
            if not row or len(row)!=columns or not row[0]:
                continue
            if is_header:
                is_header=False
                continue
            if columns==3:
                id,text,category=row
                item=(int(id),text,int(category))
            else:
                id,text=row
                item=(int(id),text)
            data.append(item)

    return data

def main(args):
    csv=args["<csv>"]
    if not is_file(csv):
        return

    clean_csv(args,csv)
    print("Done.")

if __name__ == "__main__":
    args = docopt(__doc__, version="1.0")
    main(args)
















