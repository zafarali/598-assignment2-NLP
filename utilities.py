
from nltk.stem.snowball import SnowballStemmer
STEMMER=SnowballStemmer('english')

import re, os, time, platform
from constants import *
import numpy as np
from collections import combinations_with_replacement

from string import ascii_lowercase, ascii_uppercase, digits
WHITELIST=set(ascii_lowercase+ascii_uppercase+digits+" _"+"".join(list(TOKENS.keys())))

def delete_wacky_chars(text):
    #returns text but excluding chars that aren't alphanumeric, normal punctuation, accent chars.
    def is_wacky(c):
        if c in WHITELIST:
            return 0
        a=ord(c)
        if a<48 and c not in " _#":
            return 1
        if a>=59 and a<=64:
            return 1
        if a>=92 and a<=96:
            return 1
        if a>=161 and a<=177:
            return 1
        if a>257:
            return 1
        return 0

    return "".join([c for c in text if not is_wacky(c)])

def count_tokens(ngram):
    return sum([1 if word.startswith("_") else 0 for word in ngram])

def is_caps_meaningful(previous_word,word):
    #returns true if word is capitalized and not after a period, or if it's a caps acronym
    capcount=sum([1 if c.isupper() else 0 for c in word])
    if capcount>1:
        return True
    if "_period"==previous_word:
        return False
    return capcount==1

def add_filename_prefix_to_path(prefix,source):
    #given prefix="test-" and source="/path/to/thingy.csv", returns "/path/to/test-thingy.csv
    split=source.split(os.sep)
    pieces=split[:-1]+[prefix+split[-1]]
    return os.sep.join(pieces)

class Timer:
    #easy way to show updates every X number of seconds for big jobs.
    def __init__(self,interval):
        self.start_time=time.time()
        self.last_time=self.start_time
        self.interval=interval

    def tick(self,text):
        if time.time()>self.last_time+self.interval:
            self.last_time=time.time()
            print_color(text,COLORS.YELLOW)

    def stop(self,label):
        print_color("%s took %s seconds."%(label,round(time.time()-self.start_time,1)),COLORS.YELLOW)

def get_cumulative_ngrams(words,n):
    #given words=["a","b","c","d"] and n=2
    #returns a list of all 1grams and 2grams.
    if n<2:
        return get_ngrams(words,1)
    return get_ngrams(words,n)+get_cumulative_ngrams(words,n-1)

def get_ngrams(words,n):
    if type(words) is str:
        words=re.split("[ ]+",words.strip())
    ngrams=[]
    for i,word in enumerate(words):
        if i+n>len(words):
            break
        ngrams.append(tuple(words[i:i+n]))
    return ngrams

def print_color(text,color=0,end="\n"):
    if platform.system()!="Linux":
        print(text,end=end)
    prefix=""
    if color:
        prefix+="\033[%sm"%(color-10)

    print(prefix+text+"\033[0m",end=end)

class ConfusionMatrix(object):
    def __init__(self, actual, predicted):
        """
            Creates a confusion matrix
        """
        self.classes = np.unique(actual)
        self.num_classes = len(self.classes)
        confusion_matrix = np.zeros( ( self.num_classes, self.num_classes ) )

        for i, j in combinations_with_replacement(self.classes, r=2):
        #     print('i=',i,'j=',j)
            confusion_matrix[i, j] = np.sum(np.logical_and(ACTUAL == i, PREDICT == j))
        #     print(confusion_matrix[i-1, j-1])

            confusion_matrix[j, i] = np.sum(np.logical_and(ACTUAL == j, PREDICT == i))

        self.confusion_matrix = confusion_matrix
            
    def TP(self, label=1):
        """
            Returns the True Positives for Label provided
        """
        return self.confusion_matrix[label, label]
    
    def FN(self, label=1):
        """
            Returns the False Negatives for Label Provided
        """
        trues = self.TP(label)
        false_negs = np.sum(self.confusion_matrix[label,:]) - trues
        
        return false_negs
    
    def FP(self, label=1):
        """
            Returns the False Positives for Label Provided
        """
        trues = self.TP(label)
        false_pos = np.sum(self.confusion_matrix[:,label]) - trues
        return false_pos
    
    def TN(self, label=1):
        """
            Returns the True Negatives for Label Provided
        """
        total = np.sum(self.confusion_matrix)
        total = total - self.FP(label) - self.FN(label) - self.TP(label)
        
        return total
    
    def accuracy(self, label=1):
        """
            Returns the Accuracy for the Label Provided
        """
        return (self.TP(label)+self.TN(label))/float(self.TP(label)+self.TN(label)+self.FP(label)+self.FN(label))
        
    def average_accuracy(self):
        """
            Returns the Average Accuracy for the Labels
        """
        accuracy = 0
        for label in self.classes:
            accuracy += self.accuracy(label)
        
        return accuracy/float(self.num_classes)
    
    def precision(self):
        """
            Returns the precision of the matrix
        """
        numerator, denominator = 0.0, 0.0
        
        for label in self.classes:
            numerator += self.TP(label)
            denominator += self.TP(label) + self.FP(label)
            
        return (numerator/denominator)
    
    def recall(self):
        """
            Returns the recall of the matrix
        """
        numerator, denominator= 0.0, 0.0
        
        for label in self.classes:
            numerator += self.TP(label)
            denominator += self.TP(label) + self.FN(label)
            
        return (numerator/denominator)

        
        
        