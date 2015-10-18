
# coding: utf-8



import numpy as np
import pandas as pd
import sys
from itertools import combinations_with_replacement
from utilities import ConfusionMatrix
import csv


learn_code = sys.argv[1]
test_what = sys.argv[2]

df_complete = pd.read_csv('./clean/ml_dataset_train-'+learn_code+'.csv',index_col=0)



from WordVectors import WordVectorizer
from SimpleNB import multiple_naive_bayes



np.random.seed(4)


n = len(df_complete)

train_split = 0.8


to_save = []

if test_what == 'train_size':

	to_save.append('train_size', 'accuracy', 'precision', 'recall')

	for train_size in [ 1000*n for n in range(1,5) ]:

		# train the naive bayes and obtain the actual, predicted vectors.
		actual, predicted = run_nb(train_size=train_size)

		# get confusion matrix to get metrics
		CM = ConfusionMatrix(actual, predicted)

		to_save.append( (train_size, CM.average_accuracy(), CM.precision(), CM.recall() ) )

elif test_what == 'cumulative_ngram':
	to_save.append('ngram_max', 'accuracy', 'precision', 'recall')

	for max_ngram in range(1,4):
		# train the naive bayes and obtain the actual, predicted vectors.
		actual, predicted = run_nb(ngram_range=(1,max_ngram))

		# get confusion matrix to get metrics
		CM = ConfusionMatrix(actual, predicted)

		to_save.append( (max_ngram, CM.average_accuracy(), CM.precision(), CM.recall() ) )

elif test_what == 'ngram':
	to_save.append('ngram_length', 'accuracy', 'precision', 'recall')

	for ngram_length in range(1,4):
		# train the naive bayes and obtain the actual, predicted vectors.
		actual, predicted = run_nb(ngram_range=(ngram_length,ngram_length))

		# get confusion matrix to get metrics
		CM = ConfusionMatrix(actual, predicted)

		to_save.append( (ngram_length, CM.average_accuracy(), CM.precision(), CM.recall() ) )


with open('performance_test/performance_test_'+test_what+'_'+learn_code+'.csv', 'w') as f:
	writer = csv.writer(f)
	for row in to_save:
		writer.writerow(row)



def run_nb(train_size=1000,ngram_range=(1,1),smoothing=1):

	# subset of the training set for performance purposes
	subset_indicies = np.random.choice([x for x in range(0, int(n*train_split)) ],size=train_size)
	df_train = df_complete.iloc[subset_indicies]


	# subset of testing set for performance purposes
	subset_indicies = np.random.choice([x for x in range(int(n*train_split), n) ],size=2000)
	df_test = df_complete.iloc[subset_indicies]

	# create a word vector using our training set
	wv = WordVectorizer(df_train.values, contains_prediction=True, ngram_range=ngram_range) 

	# make improvement here, convert to indicator
	X_train = wv.convert_to_word_vector(df_train.values.T[0])
	X_train[X_train > 0] = 1 
	Y_train = df_train.values.T[1]
	assert len(X_train) == len(Y_train)


	# train the model
	predictor = multiple_naive_bayes(X_train,Y_train)

	# conver to indicator
	X_test = wv.convert_to_word_vector(df_test.values.T[0])
	X_test[X_test > 0] = 1
	Y_test = df_test.values.T[1]

	# get predictions
	predicted = predictor(X_test)
	
	return Y_test, predicted


