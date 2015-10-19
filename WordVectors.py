import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2


class WordVectorizer(object):
	def __init__(self, data, contains_prediction=False, use_chi2=False, chi2_param=500, **kwargs):
		"""
			data is the training set to create the initial vocabulary
			@params:
				data: the numpy array containing our "observations" of sentences
				contains_prediction: = False: set to true if you are supplying a numpy array
									   with the predictions in the second column
				kwargs: to submit to sklearn.feature_extraction.text.CountVectorizer
			@returns:
				sparse matrix bag of words representation
		"""

		if contains_prediction:
			# transpose so that we have two rows, one with the observations and other with labels
			observations, labels = data.T 
		else:
			observations = data

		observations = map(str, observations) # converts from numpy string format to string

		self.count_vectorizer = CountVectorizer(**kwargs)
		self.bow = self.count_vectorizer.fit_transform(observations) # create vocabulary
		print(self.bow.shape)
		self.use_chi2 = False

		if use_chi2:
			assert contains_prediction==True, 'Must supply predictions as well to use chi2'
			self.ch2 = SelectKBest(chi2, k=chi2_param)
			self.ch2.fit_transform(self.bow, list(labels))
			self.use_chi2 = True



	def convert_to_word_vector(self, data, sparse=False):
		"""
			converts new data into word vectors using vocabulary
			used during initialization
			@params:
				data: the numpy array containing observations that need to be vectorized
				sparse = False: returns a sparse matrix if true, if not 
			@returns:
				numpy array of word_vectors which correspond to the data.
		"""

		to_be_returned = self.count_vectorizer.transform(data)
		to_be_returned = self.ch2.transform(to_be_returned)
		print(to_be_returned.toarray().shape)
		if not sparse:
			return to_be_returned.toarray()
		else:
			return to_be_returned

