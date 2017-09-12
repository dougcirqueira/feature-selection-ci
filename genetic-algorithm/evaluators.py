# -*- coding: iso-8859-15 -*-
'''
 Universidade Federal do Para
 Evolutionary Computing
 Student: Douglas da Rocha Cirqueira
 Professor: Claudomiro Sales
 Matricula: 201600470044

 Genetic Algorithm Framework
'''

import numpy as np
import sys
import decimal

import scipy.sparse as sp

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import svm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import string
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from collections import OrderedDict

from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from nltk import bigrams

import glob

import pprint
pp = pprint.PrettyPrinter(indent=4)

reload(sys)  
sys.setdefaultencoding('utf8')



class NeuralNetworkEvaluator(object):
	"""docstring for NeuralNetworkEvaluator"""
	def __init__(self):
		self.stemmer = PorterStemmer()
		self.wordnet_lemmatizer = WordNetLemmatizer()
		self.stopwords_PT = [word.strip() for word in open("neuralnetworkdata/stopwords_PT_R.txt").readlines()]
		self.input_final = "neuralnetworkdata/data.csv"
		self.data = pd.read_csv(self.input_final)
		self.data = self.preprocessing_PT(self.data)
		self.labels_order_2 = [1,-1]
		self.nfolds = 10
		self.skf = StratifiedKFold(n_splits=self.nfolds)

	def preprocessing_PT(self, data):
		# REMOVE URLS
		data['Message'] = data['Message'].replace(to_replace='http\\S+\\s*', value='',regex=True)

		# REMOVE hashtags
		data['Message'] = data['Message'].replace(to_replace='#\\S+', value='',regex=True)


		# REMOVE @mentions
		data['Message'] = data['Message'].replace(to_replace='@\\S+', value='',regex=True)

		# REPLACE ALL PUNCTUATION BY WHITESPACE
		data['Message'] = data['Message'].replace(to_replace='[%s]' % re.escape(string.punctuation), value=' ',regex=True)

		# To Lowercase
		data['Message'] = data['Message'].apply(lambda x: x.lower())

		# REMOVE Stopwords
		data['Message'] = data['Message'].apply(lambda x: " ".join([word for word in [item for item in x.split() if item not in self.stopwords_PT]]))

		# Stemming
		#print(data['Message'][22])
		# RSLP Stemming 
		stemmer = nltk.stem.RSLPStemmer()

		# Snowball Stemming
		#stemmer = nltk.stem.SnowballStemmer("portuguese")

		data['Message'] = data['Message'].apply(lambda x: " ".join([word for word in [stemmer.stem(item) for item in x.split() if item not in self.stopwords_PT]]))


		# REPLACE Portuguese accented characters in R with non-accented counterpart
		data['Message'] = data['Message'].apply(lambda x: unicodedata.normalize('NFKD', unicode(x)).encode('ASCII','ignore'))

		# REMOVE Numbers
		data['Message'] = data['Message'].replace(to_replace='\d+', value='',regex=True)

		return data

	def vectorizeTFIDF(self, data, ngram_range):
		if ngram_range == (1,2,3):

			ngram_range = (1,3)

			vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
			X = vectorizer.fit_transform(data)

			# TFIDF weights
			transformer = TfidfTransformer()
			X = transformer.fit_transform(X)

			return X
		

		if ngram_range == (1,3):

			vectorizer = CountVectorizer(min_df=1, ngram_range=(1,1))
			X = vectorizer.fit_transform(data)

			# TFIDF weights
			transformer = TfidfTransformer()
			X = transformer.fit_transform(X)


			ngram_range=(3,3)

			vectorizer_3 = CountVectorizer(min_df=1, ngram_range=(3,3))
			X_3 = vectorizer.fit_transform(data)


			# TFIDF weights
			transformer_3 = TfidfTransformer()
			X_3 = transformer.fit_transform(X_3)


			X_final = sp.hstack((X, X_3), format='csr')

			X = X_final

		else:
			#vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
			#X = vectorizer.fit_transform(data)

			# TFIDF weights
			#transformer = TfidfTransformer()
			#X = transformer.fit_transform(X)
			vectorizer = TfidfVectorizer()
			X = vectorizer.fit_transform(data)

			#X[:, 0] = 0

			#print X[:, 0]

		return X

	def decode(self, chromossome, X):

		indexes = [i for i,gene in enumerate(chromossome) if gene == 0]
		for index in indexes:
			X[:, index] = 0

		return X


	def evaluateWithStats(self, chromossome):
		accuracies = []
		precisions_pos = []
		precisions_neg = []
		recalls_pos = []
		recalls_neg = []
		f1_scores_pos = []
		f1_scores_neg = []
		macro_f1_vec = []
		

		data = pd.DataFrame(self.data)

		X = self.vectorizeTFIDF(self.data['Message'], ngram_range=(1,1))

		X = self.decode(chromossome, X)
		y = self.data['Truth']

		#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 10, 10, 10, 10, 10), random_state=1)
		clf = svm.SVC(kernel='linear')

		for train_idx, test_idx in self.skf.split(X, y):

			clf.fit(X[train_idx], y[train_idx])
			predicted = clf.predict(X[test_idx])

			accuracy = accuracy_score(y[test_idx], predicted)
			precision, recall, fscore, support = precision_recall_fscore_support(y[test_idx], predicted, labels=self.labels_order_2)
			macro_f1 = f1_score(y[test_idx], predicted, average='macro')

			accuracies.append(accuracy)
			precisions_pos.append(precision[0])
			precisions_neg.append(precision[1])
			recalls_pos.append(recall[0])
			recalls_neg.append(recall[1])
			f1_scores_pos.append(fscore[0])
			f1_scores_neg.append(fscore[1])
			macro_f1_vec.append(macro_f1)

		accuracyFinal = np.mean(accuracies)
		precisions_posFinal = np.mean(precisions_pos)
		precisions_negFinal = np.mean(precisions_neg)
		recalls_posFinal = np.mean(recalls_pos)
		recalls_negFinal = np.mean(recalls_neg)
		f1_scores_posFinal = np.mean(f1_scores_pos)
		f1_scores_negFinal = np.mean(f1_scores_neg)
		macro_f1Final = np.mean(macro_f1_vec)

		return (accuracyFinal, precisions_posFinal, precisions_negFinal, recalls_posFinal, recalls_negFinal,
			f1_scores_posFinal, f1_scores_negFinal, macro_f1Final)
		

class BinaryEvaluator(object):
	"""docstring for BinaryEvaluator"""
	def __init__(self):
		pass

	def decode(self, chromossome):
		return chromossome

	def evaluateWithStats(self, chromossome):

		result = sum(chromossome)
		
		return result
		


class PrisonerEvaluator(object):
	def __init__(self):
		pass
		
	def decode(self, chromossome):
		decodedChromossome = []
		for gene in chromossome:
			if gene > 0.5:
				decodedChromossome.append(("d", gene))
			elif gene < 0.5:
				decodedChromossome.append(("c", gene))

		return decodedChromossome

	def evaluateWithStats(self, chromossome, chromossomeOppositor, fitnessTable):
		decodedChm = self.decode(chromossome)
		decodedChmOppositor = self.decode(chromossomeOppositor)

		fitness = 0.
	
		cooperationCounts = 0
		cooperationGenes = []
		for geneChm, geneChmOppo in zip(decodedChm, decodedChmOppositor):
			# Pairing Calculation
			pairing = "".join([geneChm[0], geneChmOppo[0]])
			fitness += float(fitnessTable[pairing])

		fitness = fitness / len(chromossome)

		return fitness


class IntEvaluator(object):
	def __init__(self):
		self.codedGeneMax = 1
		self.codedGeneMin = 0
		self.decodedGeneMax = 130
		self.decodedGeneMin = 50
		self.decimalPrecision = 3
		

	def decode(self, chromossome):
		decodedChromossome = []
		for gene in chromossome:
			decodedGene = ((gene - self.codedGeneMin) * 
				((self.decodedGeneMax - self.decodedGeneMin)/(self.codedGeneMax-self.codedGeneMin))) + self.decodedGeneMin

			decodedGene = round(decimal.Decimal(decodedGene), self.decimalPrecision)

			decodedChromossome.append(decodedGene)

		return decodedChromossome

	def evaluate(self, chromossome, solution):
		decodedChromossome = self.decode(chromossome)

		# 1.0 / Euclidian Distance
		eucDistance = np.linalg.norm(np.array(solution) - np.array(decodedChromossome))

		if eucDistance == 0:
			eucDistance = sys.float_info.min

		result = abs(1.0 / eucDistance)
		
		return result


	def evaluateWithStats(self, chromossome, solution):
		decodedChromossome = self.decode(chromossome)

		difference = np.array(solution) - np.array(decodedChromossome)
		numMatches = np.count_nonzero(difference==0)

		# 1.0 / Euclidian Distance
		eucDistance = np.linalg.norm(difference)

		if eucDistance == 0:
			eucDistance = sys.float_info.min

		result = abs(1.0 / eucDistance)
		
		return (result, numMatches)