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
import random

class RouletteSelection(object):
	"""docstring for RouletteSelection"""
	def __init__(self):
		pass

	def selection(self, population):
		# TODO Continue
		# based on: https://stackoverflow.com/questions/177271/roulette-selection-in-genetic-algorithms
		copyPopulation = list(population)

		#sortedPopulation = sorted(copyPopulation, key=lambda x: x.getFitness(), reverse=False)
		sortedPopulation = copyPopulation

		fitnesses = [i.getFitness() for i in sortedPopulation]

		total_fitness = float(sum(fitnesses))
		rel_fitness = [f/total_fitness for f in fitnesses]
		# Generate probability intervals for each individual
		probs = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))]

		# Draw parents
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
		

class TournamentSelection(object):
	"""docstring for TournamentSelection"""
	def __init__(self):
		pass

	def selection(self, population, ringSize):
		ring = np.random.choice(population, ringSize, replace=False)
		best = sorted(ring, key=lambda x: x.getFitness(), reverse=True)[0]
		return best