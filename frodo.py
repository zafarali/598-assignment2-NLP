import math

from text_analyzer import TextAnalyzer
from constants import *
from utilities import *

class Frodo(TextAnalyzer):
    def process(self):
        if not self.silent:
            print_color("Frodo is processing data.",COLORS.GREEN)
            
        #if eager learner, run functions to process training data
        self.do_stuff()

        #don't remove this:
        self.has_processed=True

    def get_prediction_tuple(self,text):
        raise NotImplemented
        #given "text", return a 4 tuple with how confident you are for each category
        #tuple should add up to 1

        #in this example, 50% sure it's category 0, 30% for category 2, 20% category 3
        confidence=(0.5,0,0.3,0.2)
        return confidence

    def do_stuff(self):
        for id,text,category in self.training_data:
            #this will iterate over all training cases
            #the first time the for loop runs, id,text,category are set to the first example
            #the second time, id,text,category are set to the second, etc
            pass



