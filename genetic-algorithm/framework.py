# -*- coding: iso-8859-15 -*-
'''
 Universidade Federal do Para
 Evolutionary Computing
 Student: Douglas da Rocha Cirqueira
 Professor: Claudomiro Sales
 Matricula: 201600470044

 Genetic Algorithm for Prisoner's Dilemma
'''

from config import Config
from geneticalgorithm import GeneticAlgorithm
from stats import OverallStats
from evaluators import NeuralNetworkEvaluator
from os import listdir
from os.path import isfile, join
import sys
import re

import numpy as np


def main():

	resultsFilename = "results.csv"

	# General Stats Holder
	stats = OverallStats(NeuralNetworkEvaluator(), resultsFilename)

	#configs = ["%dconfig.xml" % n for n in range(1,31)]
	configs = ["1config.xml"]


	rwExecutions = []

	for config_file in configs:
		# General Configs
		config = Config(config_file)
		numOfExecutions = config.executions

		# Genetic Algorithm
		agExecutions = []
		
		for exe in range(numOfExecutions):
			ag = GeneticAlgorithm(config)
			ag.run()
			agExecutions.append(ag)
		
		#stats.computeOverallStats(agExecutions)

		stats.computeOverallStatsAll(rwExecutions, agExecutions, config, configName=config_file.split(".")[0], labels=["GA"])

	#stats.exportResults(resultsFilename)

if __name__ == '__main__':
	main()

	