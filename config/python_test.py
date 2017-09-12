import random
import numpy as np
import sys
from individuals import BinaryIndividual

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import re
import string
import unicodedata
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from collections import OrderedDict

import glob

import pprint
pp = pprint.PrettyPrinter(indent=4)

reload(sys)  
sys.setdefaultencoding('utf8')


"""
def selection(population):
		
	# TODO Continue
	# based on: https://stackoverflow.com/questions/177271/roulette-selection-in-genetic-algorithms
	copyPopulation = list(population)

	sortedPopulation = sorted(copyPopulation, key=lambda x: x.getFitness(), reverse=False)
	#sortedPopulation = copyPopulation

	fitnesses = [i.getFitness() for i in sortedPopulation]

	total_fitness = float(sum(fitnesses))
	rel_fitness = [f/total_fitness for f in fitnesses]
	# Generate probability intervals for each individual
	probs = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))]

	# Draw new population
	new_population = []

	
	next_generation_count = 0
	len_pop = len(population)

	parents = []
	while next_generation_count < len_pop:
		curr_parents = []
		for n in xrange(2):
			r = random.uniform(0,1)
			for (i, individual) in enumerate(sortedPopulation):
				if r <= probs[i]:
					curr_parents.append(individual)
					break

		parents.append((curr_parents[0], curr_parents[1]))

		next_generation_count += 2

	return parents


#fitnesses = [1,2,3,4,5,6,7,8,9,10]
#fitnesses = [4,2,3,5,8,10,1,7,9,6]
fitnesses = [1,2,3,5,20,40,60,13,200,300]
population = []

for f in fitnesses:
	i = BinaryIndividual(10, None)
	i.setFitness(f)
	population.append(i)


parents = selection(population)
"""


"""
p1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
p2 = [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]


indexes = range(0, (len(p1)+1))

#positions = sorted(np.random.choice(indexes, 2, replace=False))
positions = [3,20]
print positions

p1_slice = p1[positions[0]:positions[1]]
p2_slice = p2[positions[0]:positions[1]]

if len(p1_slice) == len(p1):
	s1 = p2_slice
	s2 = p1_slice

elif positions[0] == 0:
	s1 = p2_slice + p1[positions[1]:]
	s2 = p1_slice + p2[positions[1]:]

elif positions[1] == len(p1):
	s1 = p1[0:positions[0]] + p2_slice
	s2 = p2[0:positions[0]] + p1_slice

else:
	s1 = p1[0:positions[0]] + p2_slice + p1[positions[1]:]
	s2 = p2[0:positions[0]] + p1_slice + p2[positions[1]:]

print "S1:"
print s1

print "S2:"
print s2
"""