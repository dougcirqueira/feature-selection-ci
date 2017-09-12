# -*- coding: iso-8859-15 -*-
'''
 Universidade Federal do Para
 Evolutionary Computing
 Student: Douglas da Rocha Cirqueira
 Professor: Claudomiro Sales
 Matricula: 201600470044

 Genetic Algorithm Framework
'''

from stats import Stats
import random
from time import sleep
import numpy as np
from util import PrisonerUtil
from collections import Counter
import csv

class GeneticAlgorithm(object):
	def __init__(self, config):
		self.config = config
		self.stats = Stats()
		self.populationNumber = config.populationNumber
		self.individual = config.individualClass
		self.evaluator = config.evaluatorClass()
		self.selector = config.selectionClass()
		self.elitism = config.elitism
		self.ringSize = config.ringSize
		self.crossoverClass = config.crossoverClass()
		if config.mutator == "Gaussian":
			self.mutator = config.mutationClass(config.mutationGaussianStd, config.mutationProb)
		elif config.mutator == "BitFlip":
			self.mutator = config.mutationClass(config.mutationProb, config.bitsToMutate, config.individualSize)
		self.individualSize = config.individualSize
		self.stopCriterea = config.stopCriterea
		self.iterations = config.iterations
		self.solution = config.solution
		self.population = []
		self.toRemove = int(config.selectionFactor * self.populationNumber)
		self.mutation_prob = config.mutationProb
		self.crossover_prob = config.crossoverProb
		self.verbose = config.verbose
		self.solutionFound = False
		self.solutionFoundGeneration = None
		self.stopCriterea = config.stopCriterea
		self.iterationsConvergence = config.iterationsConvergence


	def startPopulation(self):
		population = []
		for i in range(self.populationNumber):
			sleep(0.05)
			individual = self.generateRandomIndividual()
			population.append(individual)

		return population

	def getPopulation(self):
		return self.population

	def printPopulation(self):
		for i in self.population:
			print i.getChm()
			print "==================="

	def getBestIndividual(self, population):
		return sorted(population, key=lambda x: x.getFitness(), reverse=True)[0]

	def getWorstIndividual(self, population):
		return sorted(population, key=lambda x: x.getFitness())[0]

	def replaceIndividual(self, replacement, population):
		pass

	def generateRandomIndividual(self):
		individual = self.individual(self.individualSize, None)
		return individual

	def generateNewIndividual(self, chromossome):
		individual = self.individual(self.individualSize, chromossome)
		return individual


	def calcFitness(self, individual):
		fitnessIndividual = self.evaluator.evaluateWithStats(individual.getChm())
		return fitnessIndividual

	def calcPopulationFitness(self, population):
		for individual in population:
			fitness = self.calcFitness(individual)
			individual.setFitness(fitness)

		return population
		

	def selection(self, population):
		if self.config.selection == "Roulette":
			parents = self.selector.selection(population)
		elif self.config.selection == "Tournament":
		
			parents = []

			next_generation_count = 0
			len_pop = len(population)

			while next_generation_count < len_pop:
				parent1 = self.selector.selection(population, self.ringSize)
				parent2 = self.selector.selection(population, self.ringSize)

				parents.append((parent1, parent2))

				next_generation_count += 2
		

		return parents


	def mutation(self, chromossome):
		newChromossome = self.mutator.mutate(chromossome)
		return newChromossome


	def crossover(self, parents):
		nextGeneration = []

		for parent1, parent2 in parents:
			if random.uniform(0,1) < self.crossover_prob:
				chmOffspring1, chmOffspring2 = self.crossoverClass.crossover(parent1.getChm(), parent2.getChm())
				son1 = self.generateNewIndividual(chmOffspring1)
				son2 = self.generateNewIndividual(chmOffspring2)
				nextGeneration.append(son1)
				nextGeneration.append(son2)
			else:
				nextGeneration.append(parent1)
				nextGeneration.append(parent2)

		return nextGeneration

	def runWithIterations(self):
		# Generate initial population
		self.population = self.startPopulation()

		for generation in range(self.iterations):
			# Calculate Fitness
			for individual in self.population:
				fitness = self.calcFitness(individual, self.solution)
				individual.setFitness(fitness)
				if individual.getGenesMatches() == self.individualSize and self.solutionFound == False:
				#if True and self.solutionFound == False:
					print "SOLUTION FOUND!!!"
					print "GENERATION NUMBER: %d" % generation
					print "SOLUTION MATCHES: %d" % self.getBestIndividual(self.getPopulation()).getGenesMatches()
					print "SOLUTION FITNESS: %f" % individual.getFitness()
					self.solutionFound = True
					self.solutionFoundGeneration = generation

			# Check if Prisoner Bonus is active
			if config.bonus == True:
				pass

			# Check if elitism is active
			if self.elitism and generation > 1:
				best_previous_pop = self.getBestIndividual(previous_population)
				best_current_pop = self.getBestIndividual(self.getPopulation())

				if best_previous_pop.getFitness() > best_current_pop.getFitness():
					individual_to_remove = random.choice(self.getPopulation())
					self.getPopulation().remove(individual_to_remove)
					self.getPopulation().append(best_previous_pop)

			self.stats.computeStatistics(self.getPopulation(), self.getBestIndividual(self.getPopulation()).getFitness(), self.getWorstIndividual(self.getPopulation()).getFitness(), generation, self.solutionFound, self.solutionFoundGeneration)
			
			# Selection
			parents = self.selection(self.getPopulation())

			# Crossover
			previous_population = self.getPopulation()
			self.population = self.crossover(parents)

			# Mutation
			for individual in self.population:
				individual.setChm(self.mutation(individual.getChm()))
			
			if self.verbose == True:
				print "GA Generation Number: %d" % generation
				print "Best Fitness: %f" % self.getBestIndividual(self.getPopulation()).getFitness()
				#print "Best Fitness Matches: %d" % self.getBestIndividual(self.getPopulation()).getGenesMatches()
				print "Worst Fitness: %f" % self.getWorstIndividual(self.getPopulation()).getFitness()


	def runWithConvergence(self):
		# Generate initial population
		self.population = self.startPopulation()

		lastPopMean = 0.
		currPopMean = 0.
		
		countPopMeanConv = 0
		countGeneSameBestFit = 0

		lastBestFitness = 0

		countSurpassesNeuralNet = 0

		# Save Best From each execution data
		self.data_to_output_best_chm_ga = []
		self.data_to_output_best_chm_ga.append(["generation", "fitness", "features",
			"precPosFinal","precNegFinal", "precAvg",  "recallPosFinal", 
			"recallNegFinal", "recallAvg", "f1PosFinal", "f1NegFinal", "macrof1Final", "chromossome"])

		


		for generation in range(self.iterations):

			# Calculate Fitness # TODO FIX FITNESS GOING TO 0
			self.population = self.calcPopulationFitness(self.getPopulation())


			if self.verbose == True:
				print "GA Generation Number: %d" % generation
				print "Best Fitness: %f" % self.getBestIndividual(self.getPopulation()).getFitness()
				#print "Best Fitness Matches: %d" % self.getBestIndividual(self.getPopulation()).getGenesMatches()
				print "Worst Fitness: %f" % self.getWorstIndividual(self.getPopulation()).getFitness()
				print "Mean Pop Fitness: %f" % lastPopMean
				print "----------------------------------------------------------"


			# Check if elitism is active
			if self.elitism and generation > 0:
				best_previous_pop = self.getBestIndividual(previous_population)
				best_current_pop = self.getBestIndividual(self.getPopulation())				

				if best_previous_pop.getFitness() > best_current_pop.getFitness():
					individual_to_remove = random.choice(self.getPopulation())
					self.getPopulation().remove(individual_to_remove)
					self.getPopulation().append(best_previous_pop)

			self.stats.computeStatistics(self.population, self.getBestIndividual(self.getPopulation()).getFitness(), self.getWorstIndividual(self.getPopulation()).getFitness(), generation, self.solutionFound, self.solutionFoundGeneration)


			if self.getBestIndividual(self.getPopulation()).getFitness() > 0.8073:
				i = self.getBestIndividual(self.getPopulation())
				chromossome = i.getChm()
				features = Counter(chromossome)
				self.data_to_output_best_chm_ga.append([
					generation,
					i.getFitness(), 
					features[1],
					i.precisions_posFinal,
					i.precisions_negFinal,
					(i.precisions_posFinal + i.precisions_negFinal) / 2,
					i.recalls_posFinal,
					i.recalls_negFinal,
					(i.recalls_posFinal + i.recalls_negFinal) / 2,
					i.f1_scores_posFinal,
					i.f1_scores_negFinal,
					i.macro_f1Final,
					chromossome
				])

				with open("results/%s_"%"1config" + "_neuralSurpassesBestsGA.csv", "a") as output:
					writer = csv.writer(output)
					writer.writerows(self.data_to_output_best_chm_ga)

				self.data_to_output_best_chm_ga = []				

				# Save All Population from Best Execution
				self.data_to_output_allPop = []
				self.data_to_output_allPop.append(["generation", "fitness", "features",
					"precPosFinal","precNegFinal", "precAvg",  "recallPosFinal", 
					"recallNegFinal", "recallAvg", "f1PosFinal", "f1NegFinal", "macrof1Final", "chromossome"])

				populationToExport = self.population

				for i in populationToExport:
					chromossome = i.getChm()
					features = Counter(chromossome)
					self.data_to_output_allPop.append([
						generation,
						i.getFitness(), 
						features[1],
						i.precisions_posFinal,
						i.precisions_negFinal,
						(i.precisions_posFinal + i.precisions_negFinal) / 2,
						i.recalls_posFinal,
						i.recalls_negFinal,
						(i.recalls_posFinal + i.recalls_negFinal) / 2,
						i.f1_scores_posFinal,
						i.f1_scores_negFinal,
						i.macro_f1Final,
						chromossome
					])

				with open("results/%s_"%"1config" + "neuralSurpassesAllPopulationGA.csv", "w") as output:
					writer = csv.writer(output)
					writer.writerows(self.data_to_output_allPop)

				countSurpassesNeuralNet += 1


			currBestFitness = self.getBestIndividual(self.getPopulation()).getFitness()

			countGeneSameBestFit += 1

			if currBestFitness > lastBestFitness:
				print "Zerando"
				countGeneSameBestFit = 0

			if countGeneSameBestFit == self.iterationsConvergence:
				print "CONVERGENCE"
				print "INITIAL GENERATION (%d) and END (%d)" % (generation-self.iterationsConvergence, generation)
				break
				return

			lastBestFitness = self.getBestIndividual(self.getPopulation()).getFitness()


			# Selection
			parents = self.selection(self.getPopulation())


			# Crossover
			previous_population = self.getPopulation()
			self.population = self.crossover(parents)


			# Mutation
			for individual in self.population:
				individual.setChm(self.mutation(individual.getChm()))

			if generation == self.iterations-1:
				self.population = self.calcPopulationFitness(self.getPopulation())


	def run(self):
		if self.stopCriterea == "iterations":
			self.runWithIterations()
		elif self.stopCriterea == "convergence":
			self.runWithConvergence()