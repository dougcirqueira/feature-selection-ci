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
import decimal

class BinaryIndividual(object):
	"""docstring for BinaryIndividual"""
	def __init__(self, nGenes, chromossome=None):
		self.chm = [random.randint(0,1) for i in range(nGenes)] if chromossome == None else chromossome
		self.fitness = 0.

	def setChm(self, chm):
		self.chm = chm

	def getChm(self):
		return self.chm

	def setFitness(self, fitness):
		self.fitness = fitness
	
	def getFitness(self):
		return self.fitness
		

class PrisonerIndividual(object):
	def __init__(self, nGenes, chromossome=None):
		self.chm = [random.uniform(0,1) for i in range(nGenes)] if chromossome == None else chromossome
		self.indFitness = 0.
		self.groupFitness = 0.
		self.isSolution = False
		self.genesMatches = 0

	def setChm(self, chm):
		self.chm = chm

	def getChm(self):
		return self.chm

	def setIndFitness(self, fitness):
		self.indFitness = fitness
	
	def getIndFitness(self):
		return self.indFitness

	def setGroupFitness(self, fitness):
		self.groupFitness = fitness
	
	def getGroupFitness(self):
		return self.groupFitness

	def setGenesMatches(self, genesMatches):
		self.genesMatches = genesMatches

	def getGenesMatches(self):
		return self.genesMatches


class IntIndividual(object):
	def __init__(self, nGenes, chromossome=None):
		self.chm = [random.uniform(0,1) for i in range(nGenes)] if chromossome == None else chromossome
		self.fitness = 0.
		self.isSolution = False
		self.genesMatches = 0

	def setChm(self, chm):
		self.chm = chm

	def getChm(self):
		return self.chm

	def setFitness(self, fitness):
		self.fitness = fitness
	
	def getFitness(self):
		return self.fitness

	def setGenesMatches(self, genesMatches):
		self.genesMatches = genesMatches

	def getGenesMatches(self):
		return self.genesMatches