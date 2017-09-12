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

class BitFlipMutation(object):
	"""docstring for BitFlipMutation"""
	def __init__(self, mutation_prob, bitsToMutate, lenChromossome):
		self.mutation_prob = mutation_prob
		self.numBitsToMutate = bitsToMutate
		self.lenChromossome = lenChromossome

	def mutate(self, chromossome):
		indexes = range(0, self.lenChromossome)
		positions = np.random.choice(indexes, size=self.numBitsToMutate, replace=False)
		for pos in positions:
			if random.uniform(0,1) < self.mutation_prob:
				if chromossome[pos] == 1:
					chromossome[pos] = 0
				else:
					chromossome[pos] = 1

		return chromossome
		

class GaussianMutation(object):
	"""docstring for GaussianMutation"""
	def __init__(self, gaussianMutationStd, mutation_prob):
		self.gaussianMutationStd = gaussianMutationStd
		self.mutation_prob = mutation_prob
		
	def mutate(self, chromossome):
		newChromossome = []
		for gene in chromossome:
			# Mean is the gene value, std from config file
			newGene = random.gauss(gene, self.gaussianMutationStd)

			if newGene > 1:
				newGene = 1.0
			elif newGene < 0:
				newGene = 0.0

			newChromossome.append(newGene)

		return newChromossome


	def mutateMeanZero(self, chromossome):
		newChromossome = []
		for gene in chromossome:
			# Mean is zero, std from config file
			newGene = gene + random.gauss(0, self.gaussianMutationStd)

			if newGene > 1:
				newGene = 1.0
			elif newGene < 0:
				newGene = 0.0

			newChromossome.append(newGene)

		return newChromossome

	def mutateGa(self, chromossome):
		newChromossome = []
		for gene in chromossome:

			if random.uniform(0,1) < self.mutation_prob:
				# Mean is the gene value, std from config file
				newGene = random.gauss(gene, self.gaussianMutationStd)

				if newGene > 1:
					newGene = 1.0
				elif newGene < 0:
					newGene = 0.0

				newChromossome.append(newGene)
			else:
				newChromossome.append(gene)

		return newChromossome
		
	def mutateReflexionGA(self, chromossome):
		newChromossome = []
		for gene in chromossome:

			if random.uniform(0,1) < self.mutation_prob:
				# Mean is the gene value, std from config file
				newGene = random.gauss(gene, self.gaussianMutationStd)

				if newGene > 1:
					newGene = 1 - (1 - newGene)
				elif newGene < 0:
					newGene = abs(newGene)

				newChromossome.append(newGene)
			else:
				newChromossome.append(gene)

		return newChromossome


	def mutateReflexionGAMeanZero(self, chromossome):
		newChromossome = []
		for gene in chromossome:

			if random.uniform(0,1) < self.mutation_prob:
				# Mean is zero, std from config file
				newGene = gene + random.gauss(0, self.gaussianMutationStd)

				if newGene > 1:
					newGene = 1 - (1 - newGene)
				elif newGene < 0:
					newGene = abs(newGene)

				newChromossome.append(newGene)
			else:
				newChromossome.append(gene)

		return newChromossome
	