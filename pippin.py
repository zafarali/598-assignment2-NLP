import math

from text_analyzer import TextAnalyzer
from constants import *
from utilities import *

class Pippin(TextAnalyzer):
    def process(self,k=5,ngram_max=2):
        self.ngram_max=ngram_max
        self.k=k

        if not self.silent:
            print_color("Pippin is processing data.",COLORS.GREEN)
            
        self.process_training_data()

        self.has_processed=True

    def process_training_data(self):
        #where keys are ngrams, values are a set of all ids with that ngram
        self.ngram_count={}
        self.categories={}
        
        counter=0
        timer=Timer(self.interval)
        for id,text,category in self.training_data:
            counter+=1
            timer.tick("Processing item %s/%s"%(counter,len(self.training_data)))
            
            grams=get_cumulative_ngrams(text,self.ngram_max)
            self.categories[id]=category
            
            for gram in grams:
                if gram not in self.ngram_count:
                    self.ngram_count[gram]=set()
                self.ngram_count[gram].add(id)

    def get_prediction_tuple(self,text):
        ngrams=set(get_cumulative_ngrams(text,self.ngram_max))
        neighbour_points={}
        for ngram in ngrams:
            if ngram in self.ngram_count:
                for id in self.ngram_count[ngram]:
                    if id not in neighbour_points:
                        neighbour_points[id]=0
                    neighbour_points[id]+=1
        
        highest=[(neighbour_points[id],id) for id in neighbour_points]
        highest=sorted(highest)[-self.k:]
        cats=[self.categories[id] for score,id in highest]
        confidence=tuple([cats.count(i)/self.k for i in range(4)])
        return confidence

