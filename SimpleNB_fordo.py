import math

from text_analyzer import TextAnalyzer
from constants import *
from utilities import *


import numpy as np
import pandas as pd
from SimpleNB import multiple_naive_bayes
from WordVectors import WordVectorizer



class NB(TextAnalyzer):
    def process(self, use_chi2=True, chi2_param=5000):
        if not self.silent:
            print_color("Frodo is processing data.",COLORS.GREEN)
            
        #if eager learner, run functions to process training data
        # self.do_stuff()

        # df_complete = pd.read_csv(self.source_csv, index_col=0)

        # np.random.seed(4)
        # subset_indicies = np.random.choice([x for x in range(0,len(df_complete))],size=15000)
        # df_train = df_complete.iloc[subset_indicies]
        # self.df_train = df_train

        training_data = np.array(self.training_data[:1000])
        
        interview_text = training_data[:,1]
        Y = training_data[:,2].astype(int)
        df_train = np.array([interview_text, Y]).T

        # training to be done
        self.wv = WordVectorizer(df_train, contains_prediction=True, use_chi2=use_chi2, chi2_param=chi2_param)
        X = self.wv.convert_to_word_vector(df_train[:,0])
        X[X > 0] = 1
        # Y = df_train[:,1]
        # assert len(X) == len(Y), str(X.shape)+' is not equal to '+str(Y.shape)
        # print(df_train)
        # print(X.shape, Y.shape)
        # print(df_train.shape)

        # internalpurposes
        self.X_internal = X
        self.Y_internal = Y
        self.df_train_internal = df_train
        self.interview_text_internal = interview_text

        try:
            self.predictor = multiple_naive_bayes(X,Y)
        except Exception as e:
            print(str(e))
        #don't remove this:
        self.has_processed=True

    def get_prediction_tuple(self,text):
        X = self.wv.convert_to_word_vector([text])
        X[X > 0] = 1
        np_thing= self.predictor(X, only_probabilities=True, normalize=True)
        return np_thing.tolist()[0]

        # raise NotImplemented
        #given "text", return a 4 tuple with how confident you are for each category
        #tuple should add up to 1

        #in this example, 50% sure it's category 0, 30% for category 2, 20% category 3
        # confidence=(0.5,0,0.3,0.2)
        # return confidence

    def do_stuff(self):
        for id,text,category in self.training_data:
            #this will iterate over all training cases
            #the first time the for loop runs, id,text,category are set to the first example
            #the second time, id,text,category are set to the second, etc
            pass



