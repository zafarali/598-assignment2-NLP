import math

from text_analyzer import TextAnalyzer
from constants import *
from utilities import *


import numpy as np
import pandas as pd
from SimpleNB import multiple_naive_bayes
from WordVectors import WordVectorizer



class NB(TextAnalyzer):
    def process(self, use_chi2=True, chi2_param=5000,nb_max=1000):
        if not self.silent:
            print_color("Frodo is processing data.",COLORS.GREEN)
            
        training_data = np.array(self.training_data[:nb_max])
        
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
