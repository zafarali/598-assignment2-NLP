import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

class WordVectorizer(object):
	def __init__(self, data, contains_prediction=False, **kwargs):
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

		self.vectorizer = CountVectorizer(**kwargs)
		bow = self.vectorizer.fit_transform(observations) # create vocabulary

		return bow

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
		if not sparse:
			return self.vectorizer.transform(data).toarray()
		else:
			return self.vectorizer.transform(data)

