
# coding: utf-8
import numpy as np
import pandas as pd
import sys
from itertools import combinations_with_replacement, product
from utilities import ConfusionMatrix
import csv


learn_code = sys.argv[1]
test_what = sys.argv[2]




from WordVectors import WordVectorizer
from SimpleNB import multiple_naive_bayes



np.random.seed(4)




def run_nb(train_size=2000,ngram_range=(1,1),smoothing=1, learn_code='0000', kbest=None):

	df_complete = pd.read_csv('./clean/ml_dataset_train-'+learn_code+'.csv',index_col=0)

	n = len(df_complete)

	train_split = 0.8

	# subset of the training set for performance purposes
	subset_indicies = np.random.choice([x for x in range(0, int(n*train_split)) ],size=train_size)
	df_train = df_complete.iloc[subset_indicies]


	# subset of testing set for performance purposes
	subset_indicies = np.random.choice([x for x in range(int(n*train_split), n) ],size=2000)
	df_test = df_complete.iloc[subset_indicies]

	# create a word vector using our training set
	if kbest:
		wv = WordVectorizer(df_train.values, contains_prediction=True, ngram_range=ngram_range, use_chi2=True, chi2_param=kbest) 
	else:
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


to_save = []

if test_what == 'train_size':

	to_save.append(('train_size', 'accuracy', 'precision', 'recall'))

	for train_size in [ 1000*n for n in range(1,10) ]:
		print('train_size:',train_size)
		# train the naive bayes and obtain the actual, predicted vectors.
		actual, predicted = run_nb(train_size=train_size, learn_code=learn_code)

		# get confusion matrix to get metrics
		CM = ConfusionMatrix(actual, predicted)

		to_save.append( (train_size, CM.average_accuracy(), CM.precision(), CM.recall() ) )

elif test_what == 'cumulative_ngram':
	to_save.append(('ngram_max', 'accuracy', 'precision', 'recall'))

	for max_ngram in range(4,9):
		print('max_ngram:',max_ngram)
		# train the naive bayes and obtain the actual, predicted vectors.
		actual, predicted = run_nb(ngram_range=(1,max_ngram),  learn_code=learn_code)

		# get confusion matrix to get metrics
		CM = ConfusionMatrix(actual, predicted)

		to_save.append( (max_ngram, CM.average_accuracy(), CM.precision(), CM.recall() ) )

elif test_what == 'ngram':
	to_save.append(('ngram_length', 'accuracy', 'precision', 'recall'))

	for ngram_length in range(1,9):
		print('ngram_length:',ngram_length)
		# train the naive bayes and obtain the actual, predicted vectors.
		actual, predicted = run_nb(ngram_range=(ngram_length,ngram_length),  learn_code=learn_code)

		# get confusion matrix to get metrics
		CM = ConfusionMatrix(actual, predicted)

		to_save.append( (ngram_length, CM.average_accuracy(), CM.precision(), CM.recall() ) )

elif test_what == 'learn_codes':
	to_save.append(('learn_code', 'accuracy', 'precision', 'recall'))

	for combo in map(lambda x: ''.join(map(str, x)), product([0,1], repeat=4)):
		print('combo:',combo)
		actual, predicted = run_nb(learn_code=combo, train_size=6000, ngram_range=(1,1))

		CM = ConfusionMatrix(actual, predicted)

		to_save.append( (combo, CM.average_accuracy(), CM.precision(), CM.recall()) )
elif test_what == 'chi2':
	to_save.append(['kbest', 'accuracy', 'precision', 'recall'])

	for kbest in [ 1000*n for n in range(1,30)]:
		print('kbest:',kbest)
		actual, predicted = run_nb(kbest=kbest, train_size=10000, ngram_range=(1,1))

		CM = ConfusionMatrix(actual, predicted)

		to_save.append( (kbest, CM.average_accuracy(), CM.precision(), CM.recall()) )



with open('performance_test/performance_test'+test_what+'_'+learn_code+'.csv', 'w') as f:
	writer = csv.writer(f)
	for row in to_save:
		writer.writerow(row)


