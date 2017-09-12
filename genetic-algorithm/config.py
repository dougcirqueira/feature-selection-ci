# -*- coding: iso-8859-15 -*-
'''
 Universidade Federal do Para
 Evolutionary Computing
 Student: Douglas da Rocha Cirqueira
 Professor: Claudomiro Sales
 Matricula: 201600470044

 Genetic Algorithm Framework
'''

import xml.etree.ElementTree as ET

from individuals import IntIndividual
from individuals import PrisonerIndividual
from individuals import BinaryIndividual
from individuals import NeuralNetIndividual

from evaluators import IntEvaluator
from evaluators import PrisonerEvaluator
from evaluators import BinaryEvaluator
from evaluators import NeuralNetworkEvaluator

from mutations import GaussianMutation
from mutations import BitFlipMutation

from crossover import ArithmeticCrossover
from crossover import TwoPointCrossover
from crossover import UniformCrossover

from selection import TournamentSelection
from selection import RouletteSelection

import csv

class Config(object):
	def __init__(self, config_file):
		""" This method reads all configs from the config.xml file.
		"""
		try:
			tree = ET.parse('config/' + config_file)
		except Exception:
			raise Exception("read_config: Invalid or Nonexistent config.xml file.")
		
		root = tree.getroot()

		#try:
		self.executions = int(root.find('general').find('executions').text)

		self.populationNumber = int(root.find('general').find('populationNumber').text)

		self.individual = root.find('general').find('individual').text
		if self.individual == "IntIndividual":
			self.individualClass = IntIndividual
		elif self.individual == "PrisonerIndividual":
			self.individualClass = IntIndividual
		elif self.individual == "BinaryIndividual":
			self.individualClass = BinaryIndividual
		elif self.individual == "NeuralNetIndividual":
			self.individualClass = NeuralNetIndividual


		self.evaluator = root.find('general').find('evaluator').text
		if self.evaluator == "IntEvaluator":
			self.evaluatorClass = IntEvaluator
		elif self.evaluator == "PrisonerEvaluator":
			self.evaluatorClass = PrisonerEvaluator
		elif self.evaluator == "BinaryEvaluator":
			self.evaluatorClass = BinaryEvaluator
		elif self.evaluator == "NeuralNetworkEvaluator":
			self.evaluatorClass = NeuralNetworkEvaluator

		self.individualSize = int(root.find('general').find('indiviualSize').text)

		self.stopCriterea = root.find('general').find('stopCriterea').text

		self.iterations = int(root.find('general').find('iterations').text)

		self.iterationsConvergence = int(root.find('general').find('iterationsConvergence').text)		
		
		self.solutionValue = root.find('solution').find('value').text
		self.solution = [float(val) for val in self.solutionValue.split(",")]
		
		self.selectionFactor = float(root.find('general').find('selectionFactor').text)

		self.verbose = bool(root.find('general').find('verbose').text)

		self.ringSize = int(root.find('general').find('ringSize').text)

		self.selection = root.find('general').find('selectionClass').text
		if self.selection == "Tournament":
			self.selectionClass = TournamentSelection
		elif self.selection == "Roulette":
			self.selectionClass = RouletteSelection

		self.elitism = bool(root.find('general').find('elitism').text)			

		self.crossoverProb = float(root.find('general').find('crossoverProb').text)

		self.crossover = root.find('general').find('crossoverClass').text
		if self.crossover == "Arithmetic":
			self.crossoverClass = ArithmeticCrossover
		elif self.crossover == "TwoPoint":
			self.crossoverClass = TwoPointCrossover
		elif self.crossover == "Uniform":
			self.crossoverClass = UniformCrossover


		self.mutationProb = float(root.find('general').find('mutationProb').text)

		self.mutator = root.find('general').find('mutationClass').text
		if self.mutator == "Gaussian":
			self.mutationClass = GaussianMutation
		elif self.mutator == "BitFlip":
			self.mutationClass = BitFlipMutation
			self.bitsToMutate = int(root.find('prisonerSettings').find('cooperationSize').text)
		
		self.mutationGaussianStd = float(root.find('general').find('bitsToMutate').text)

		
		problem = root.find('problem').text
		if problem == "prisoner":
			self.fitnessType = root.find('prisonerSettings').find('fitnessType').text

			self.comparisonSize = int(root.find('prisonerSettings').find('comparisonSize').text)
			print self.comparisonSize
			self.bonus = bool(root.find('prisonerSettings').find('bonus').text)
			self.cooperationSize = int(root.find('prisonerSettings').find('cooperationSize').text)
			self.bonusMin = int(root.find('prisonerSettings').find('bonusMin').text)
			self.bonusMax = int(root.find('prisonerSettings').find('bonusMax').text)
			fitnessTableFile = root.find('prisonerSettings').find('fitnessTableFile').text

			print fitnessTableFile

			self.fitnessTableFromFile = {"individual": {}, "group": {}}



			with open('config/' + fitnessTableFile, mode='r') as infile:
			    print "fine"
			    reader = csv.DictReader(infile)
			    
			    for row in reader:
			    	self.fitnessTableFromFile[row["fitType"]]["dc"] = row["dc"]
			    	self.fitnessTableFromFile[row["fitType"]]["cc"] = row["cc"]
			    	self.fitnessTableFromFile[row["fitType"]]["dd"] = row["dd"]
			    	self.fitnessTableFromFile[row["fitType"]]["cd"] = row["cd"]

			if self.fitnessType == "individual":
				self.fitnessTable = self.fitnessTableFromFile["individual"]
			elif self.fitnessType == "group":
				self.fitnessTable = self.fitnessTableFromFile["group"]

		#except Exception:
		#	raise Exception("read_config: Invalid config.xml file.")
