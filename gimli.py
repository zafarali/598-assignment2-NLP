import math

from text_analyzer import TextAnalyzer
from constants import *
from utilities import *

class Gimli (TextAnalyzer):
    def process(self,ngram_max=2,filter_threshold=1):
        self.ngram_max=ngram_max
        self.filter_threshold=filter_threshold

        if not self.silent:
            print_color("Processing data.",COLORS.GREEN)
        self.process_training_data()
        if not self.silent:
            print_color("Scoring ngrams.",COLORS.GREEN)
        self.score_ngrams()

        self.has_processed=True

    def get_prediction_tuple(self,text):
        ngrams=get_cumulative_ngrams(text,self.ngram_max)
        confidence=[0 for i in range(OPTION_COUNT)]

        for ngram in ngrams:
            if ngram not in self.ngram_scores or count_tokens(ngram)>0:
                continue
            scores=self.ngram_scores[ngram]
            confidence=[confidence[i]+scores[i]**3 for i in range(OPTION_COUNT)]

        total=sum(confidence)
        total=total if total else 1
        confidence=[i/total for i in confidence]
        return confidence

    def process_training_data(self):
        #where keys are ngrams, values are (0,0,0,0) for number of times they occur for each category
        self.ngram_count={}
        #each number is how many ngrams were processed for that category in total
        self.ngram_total=[0 for i in range(OPTION_COUNT)]

        counter=0
        timer=Timer(self.interval)
        for id,text,category in self.training_data:
            counter+=1
            timer.tick("Processing item %s/%s"%(counter,len(self.training_data)))

            grams=get_cumulative_ngrams(text,self.ngram_max)

            for gram in grams:
                if gram not in self.ngram_count:
                    self.ngram_count[gram]=[0 for i in range(OPTION_COUNT)]
                count=self.ngram_count[gram]
                count[category]+=1
                self.ngram_count[gram]=count

        #filter out ngrams with only 'filter threshold' number of occurences
        if self.filter_threshold:
            self.ngram_count={key:self.ngram_count[key] for key in self.ngram_count if sum(self.ngram_count[key])>self.filter_threshold}

        for category in range(OPTION_COUNT):
            self.ngram_total[category]=sum([self.ngram_count[ngram][category] for ngram in self.ngram_count])

    def score_ngrams(self):
        #where keys are ngrams, values are (0,0,0,0) which add up to 1, meaning belief in each category
        self.ngram_scores={}
        normalized=[self.ngram_total[i]/max(self.ngram_total) for i in range(OPTION_COUNT)]

        i=0
        timer=Timer(self.interval)
        for ngram in self.ngram_count:
            i+=1
            timer.tick("Scoring item %s/%s"%(i,len(self.ngram_count)))

            count=self.ngram_count[ngram]
            adjusted_count=[0 if not normalized[i] else count[i]/normalized[i] for i in range(OPTION_COUNT)]
            total=sum(adjusted_count)
            scores=[adjusted_count[i]/total for i in range(OPTION_COUNT)]
            self.ngram_scores[ngram]=scores




