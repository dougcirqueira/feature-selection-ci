# -*- coding: iso-8859-15 -*-
'''
 Universidade Federal do Para
 Evolutionary Computing
 Student: Douglas da Rocha Cirqueira
 Professor: Claudomiro Sales
 Matricula: 201600470044

 Genetic Algorithm Framework
'''

import random
import numpy as np
import decimal

class TwoPointCrossover(object):
	"""docstring for TwoPointCrossover"""
	def __init__(self):
		pass

	def crossover(self, chmParent1, chmParent2):
		indexes = range(0, len(chmParent1))
		positions = sorted(np.random.choice(indexes, 2, replace=False))

		p1_slice = chmParent1[positions[0]:positions[1]]
		p2_slice = chmParent2[positions[0]:positions[1]]

		if len(p1_slice) == len(chmParent1):
			offspring1 = p2_slice
			offspring2 = p1_slice

		elif positions[0] == 0:
			offspring1 = p2_slice + chmParent1[positions[1]:]
			offspring2 = p1_slice + chmParent2[positions[1]:]

		elif positions[1] == len(chmParent1):
			offspring1 = chmParent1[0:positions[0]] + p2_slice
			offspring2 = chmParent2[0:positions[0]] + p1_slice

		else:
			offspring1 = chmParent1[0:positions[0]] + p2_slice + chmParent1[positions[1]:]
			offspring2 = chmParent2[0:positions[0]] + p1_slice + chmParent2[positions[1]:]

		return (offspring1, offspring2)


class UniformCrossover(object):
	"""docstring for UniformCrossover"""
	def __init__(self):
		pass

	def crossover(self, chmParent1, chmParent2):
		offspring1 = chmParent1
		offspring2 = chmParent2

		for pos in xrange(len(chmParent1)):
			geneP1 = chmParent1[pos]
			geneP2 = chmParent2[pos]
			if chmParent1[pos] != chmParent2[pos]:
				if random.uniform(0,1) < 0.5:
					offspring1[pos] = geneP2
					offspring2[pos] = geneP1

		return (offspring1, offspring2)
		

class ArithmeticCrossover(object):
	"""docstring for Crossover"""
	def __init__(self):
		pass

	def crossover(self, chmParent1, chmParent2):
		a_fac = random.uniform(0,1)

		offspring1 = ((a_fac * np.array(chmParent1)) + ((1 - a_fac) * np.array(chmParent2)))
		offspring2 = (((1 - a_fac) * np.array(chmParent1)) + (a_fac * np.array(chmParent2)))

		return (offspring1, offspring2)