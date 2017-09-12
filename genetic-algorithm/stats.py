# -*- coding: iso-8859-15 -*-
'''
 Universidade Federal do Para
 Evolutionary Computing
 Student: Douglas da Rocha Cirqueira
 Professor: Claudomiro Sales
 Matricula: 201600470044

 Genetic Algorithm Framework
'''

import matplotlib.pyplot as matplot
import numpy as np
import csv
import sys
from evaluators import NeuralNetworkEvaluator
from collections import Counter

class Stats(object):
	def __init__(self):
		self.population = []
		self.avgFitnessVec = []
		self.bestFitVec = []
		self.worstFitVec = []
		self.avgGenesMatchesVec = []
		self.generation = 0
		self.solutionFound = False
		self.solutionFoundGeneration = None
		self.evaluator = NeuralNetworkEvaluator()
		self.finalPercentCs = 0
		self.finalPercentDs = 0

	def computeStatistics(self, population, best, worst, generation, solutionFound, solutionFoundGeneration):

		self.population = population

		avgFitness = sum([i.getFitness() for i in population]) / len(population)
		avgGeneMatches = round(sum([i.getGenesMatches() for i in population]) / len(population))

		self.avgFitnessVec.append(avgFitness)
		self.bestFitVec.append(best)
		self.worstFitVec.append(worst)
		self.avgGenesMatchesVec.append(avgGeneMatches)
		self.bestIndividual = sorted(population, key=lambda x: x.getFitness(), reverse=True)[0]
		self.generation = generation
		self.solutionFound = solutionFound
		self.solutionFoundGeneration = solutionFoundGeneration


	def getAvgFitnessVec(self):
		return self.avgFitnessVec

	def getBestFitnessVec(self):
		return self.bestFitVec

	def getWorstFitnessVec(self):
		return self.worstFitVec

	def getAvgGenesMatchesVec(self):
		return self.avgGenesMatchesVec

	def getBestIndividual(self):
		return self.bestIndividual

	def plotAvgGeneMatches(self):
		x_axis = np.arange(len(self.avgGenesMatchesVec))

		# Genes Matches plot
		fit_graph = matplot
		fit_graph.gca().set_color_cycle(['green'])

		fit_graph.plot(x_axis, self.avgGenesMatchesVec)

		fit_graph.legend(['Avg'], loc='upper_left')
		fit_graph.suptitle('Average Genes Match')
		fit_graph.xlabel('Generations')
		fit_graph.ylabel('Avg Genes Matches')
		fit_graph.show()


	def plotAvgPopFitness(self):
		x_axis = np.arange(len(self.avgFitnessVec))

		# Fitness plot
		fit_graph = matplot
		fit_graph.gca().set_color_cycle(['green', 'blue', 'red'])

		fit_graph.plot(x_axis, self.bestFitVec)
		fit_graph.plot(x_axis, self.avgFitnessVec)
		fit_graph.plot(x_axis, self.worstFitVec)
		
		fit_graph.legend(['Best', 'Avg', 'Worst'], loc='upper_left')
		fit_graph.suptitle('Fitness')
		fit_graph.xlabel('Generations')
		fit_graph.ylabel('Fitness')
		fit_graph.show()

class OverallStats(object):
	"""docstring for OverallStats"""
	def __init__(self, evaluator, filename):
		# TODO Get from config file automatically
		self.evaluator = evaluator
		self.filename = filename
		self.data_to_output = []
		self.data_to_output.append(["config",
			"algorithm", 
			"populationSize", 
			"individualSize", 
			"precision", 
			"accuracy",
			"convergenceIteration",
			"iterations"])

		with open("results/" + self.filename, "a") as output:
			writer = csv.writer(output)
			writer.writerows(self.data_to_output)


	def exportResults(self, filename):
		with open("results/" + filename, "w") as output:
		    writer = csv.writer(output)
		    writer.writerows(self.data_to_output)

		with open("results/" + filename + "_bestsRW.csv", "w") as output:
		    writer = csv.writer(output)
		    writer.writerows(self.data_to_output_best_chm_rw)

		with open("results/" + self.filename + "_bestsGA.csv", "w") as output:
		    writer = csv.writer(output)
		    writer.writerows(self.data_to_output_best_chm_ga)

	def safeLog2(self, val):
		if val <= 0:
			return 0.0000000001

		return np.log2(val)


	def computeOverallStatsAll(self, executions_RW, executions_GA, config, configName, labels):

		# TODO Use variable labels

		# By default
		minLenRW = config.iterations
		minLenGA = config.iterations

		if executions_GA:
			# Genetic Algorithm preparation
			avgBestFitnessExecs_GA = []
			avgWorstFitnessExecs_GA = []
			avgFitnessVecExecs_GA = []
			avgGenesMatchVecOfVecs_GA = []

			bestIndividualVec_GA = []
			bestIndividualMatches_GA = []
			solutionFound_GA = False
			iterations_GA = executions_GA[0].stats.generation
			maxMatches_GA = 0
			solutionFoundGeneration_GA = None

			meanGenerationConv = 0

			exeIndex = 0
			for exe in executions_GA:
				# List of Lists (len = shortest number of generations in executions)
				avgBestFitnessExecs_GA.append(exe.stats.getBestFitnessVec())
				avgWorstFitnessExecs_GA.append(exe.stats.getWorstFitnessVec())
				avgFitnessVecExecs_GA.append(exe.stats.getAvgFitnessVec())
				avgGenesMatchVecOfVecs_GA.append(exe.stats.getAvgGenesMatchesVec())

				# List of bests from each execution (len = number of executions)
				bestIndividualVec_GA.append(exe.stats.getBestIndividual())
				bestIndividualMatches_GA.append(exe.stats.getBestIndividual().getGenesMatches())
				if exe.stats.solutionFound == True:
					# Solution found
					solutionFound_GA = True
					# Iterations
					iterations_GA = exe.stats.generation
					# Solution found generation
					solutionFoundGeneration_GA = exe.stats.solutionFoundGeneration

				meanGenerationConv += exe.stats.generation

				exeIndex += 1

			meanGenerationConv = ((1. * meanGenerationConv) / len(executions_GA)) + 1

			minLenGA = min(len(l) for l in avgBestFitnessExecs_GA)


		if executions_RW:
			# Random Walk preparation
			avgBestFitnessExecs_RW = []
			avgWorstFitnessExecs_RW = []
			avgFitnessVecExecs_RW = []
			avgGenesMatchVecOfVecs_RW = []
			bestIndividualVec_RW = []
			bestIndividualMatches_RW = []
			solutionFound_RW = False
			iterations_RW = executions_RW[0].stats.generation
			maxMatches_RW = 0
			solutionFoundGeneration_RW = None

			for exe in executions_RW:
				# List of Lists (len = longest number of generations in executions)
				avgBestFitnessExecs_RW.append(exe.stats.getBestFitnessVec())
				avgWorstFitnessExecs_RW.append(exe.stats.getWorstFitnessVec())
				avgFitnessVecExecs_RW.append(exe.stats.getAvgFitnessVec())
				avgGenesMatchVecOfVecs_RW.append(exe.stats.getAvgGenesMatchesVec())
				# List of bests from each execution (len = number of executions)
				bestIndividualVec_RW.append(exe.stats.getBestIndividual())
				bestIndividualMatches_RW.append(exe.stats.getBestIndividual().getGenesMatches())
				if exe.stats.solutionFound == True:
					# Solution found
					solutionFound_RW = True
					# Iterations
					iterations_RW = exe.stats.generation
					# Solution found generation
					solutionFoundGeneration_RW = exe.stats.solutionFoundGeneration

			minLenRW = min(len(l) for l in avgBestFitnessExecs_RW)


		minLen = min(minLenRW, minLenGA)


		if executions_GA:
			if config.stopCriterea == "convergence":
				print "MAX LEN GA: ", max(len(l) for l in avgBestFitnessExecs_GA)

				for i in range(config.executions):
					avgBestFitnessExecs_GA[i] = avgBestFitnessExecs_GA[i][len(avgBestFitnessExecs_GA[i]) - minLen:]
					avgWorstFitnessExecs_GA[i] = avgWorstFitnessExecs_GA[i][len(avgWorstFitnessExecs_GA[i]) - minLen:]
					avgFitnessVecExecs_GA[i] = avgFitnessVecExecs_GA[i][len(avgFitnessVecExecs_GA[i]) - minLen:]
					avgGenesMatchVecOfVecs_GA[i] = avgGenesMatchVecOfVecs_GA[i][len(avgGenesMatchVecOfVecs_GA[i]) - minLen:]

				print "MAX LEN GA AFTER TRUNCATE: ", max(len(l) for l in avgBestFitnessExecs_GA)

			# From List to Numpy Array
			avgBestFitnessExecsArray_GA = np.array(avgBestFitnessExecs_GA)
			avgWorstFitnessExecsArray_GA = np.array(avgWorstFitnessExecs_GA)
			avgFitnessVecExecsArray_GA = np.array(avgFitnessVecExecs_GA)
			avgGenesMatchVecOfVecsArray_GA = np.array(avgGenesMatchVecOfVecs_GA)

			# (len = number of executions)
			#bestIndividualChmArray_GA = np.array([self.evaluator.decode(i.getChm()) for i in bestIndividualVec_GA])
			bestIndividualChmArray_GA = np.array([i.getChm() for i in bestIndividualVec_GA])
			bestIndividualMatchesArray_GA = np.array(bestIndividualMatches_GA)

			# Max Matches
			maxMatches_GA = max(bestIndividualMatchesArray_GA)

			finalMeanBest_GA = np.mean(avgBestFitnessExecsArray_GA, axis=0)
			finalMeanWorst_GA = np.mean(avgWorstFitnessExecsArray_GA, axis=0)
			finalMeanAvg_GA = np.mean(avgFitnessVecExecsArray_GA, axis=0)
			finalMeanGeneMatches_GA = np.mean(avgGenesMatchVecOfVecsArray_GA, axis=0)

			finalStdBest_GA = np.std(avgBestFitnessExecsArray_GA, axis=0)
			finalStdWorst_GA = np.std(avgWorstFitnessExecsArray_GA, axis=0)
			finalStdAvg_GA = np.std(avgFitnessVecExecsArray_GA, axis=0)
			#finalStdAvg_GA = np.array([0.03, 0.09])
			finalStdGeneMatches_GA = np.std(avgGenesMatchVecOfVecsArray_GA, axis=0)

			# Precision
			#print bestIndividualChmArray_GA
			#sys.exit()
			#stdGenesBestInd_GA = np.std(bestIndividualChmArray_GA, axis=0)
			#precision_GA = np.mean(stdGenesBestInd_GA)
			precision_GA = 0

			# Average Matches
			avgMatches_GA = np.mean(bestIndividualMatchesArray_GA)

			# Std Matches
			stdMatches_GA = np.std(bestIndividualMatchesArray_GA)

			# Accuracy
			#accuracy_GA = (avgMatches_GA * 100) / len(bestIndividualVec_GA[0].getChm())
			accuracy_GA = 0.
			for i in bestIndividualVec_GA:
				accuracy_GA += i.getFitness()

			accuracy_GA = (1. * accuracy_GA) / len(bestIndividualVec_GA)

			# Save General Data
			self.data_to_output = []
			self.data_to_output.append([configName, "ga", config.populationNumber, config.individualSize, precision_GA,
			 accuracy_GA, meanGenerationConv, iterations_GA+1])

			with open("results/" + self.filename, "a") as output:
				writer = csv.writer(output)
				writer.writerows(self.data_to_output)

			# Save Best From each execution data
			self.data_to_output_best_chm_ga = []
			self.data_to_output_best_chm_ga.append(["fitness", "features",
				"precPosFinal","precNegFinal", "precAvg",  "recallPosFinal", 
				"recallNegFinal", "recallAvg", "f1PosFinal", "f1NegFinal", "macrof1Final", "chromossome"])

			for i in bestIndividualVec_GA:
				chromossome = i.getChm()
				features = Counter(chromossome)
				self.data_to_output_best_chm_ga.append([
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

			with open("results/%s_"%configName + "bestsGA.csv", "w") as output:
				writer = csv.writer(output)
				writer.writerows(self.data_to_output_best_chm_ga)

			# Save All Population from Best Execution
			bestFromExecutions = sorted(bestIndividualVec_GA, key=lambda x: x.getFitness(), reverse=True)[0]
			indexBest = bestIndividualVec_GA.index(bestFromExecutions)

			populationToExport = executions_GA[indexBest].population

			self.data_to_output_allPop = []
			self.data_to_output_allPop.append(["fitness", "features",
				"precPosFinal","precNegFinal", "precAvg",  "recallPosFinal", 
				"recallNegFinal", "recallAvg", "f1PosFinal", "f1NegFinal", "macrof1Final", "chromossome"])

			for i in populationToExport:
				chromossome = i.getChm()
				features = Counter(chromossome)
				self.data_to_output_allPop.append([
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

			with open("results/%s_"%configName + "allPopulationGA.csv", "w") as output:
				writer = csv.writer(output)
				writer.writerows(self.data_to_output_allPop)



		if executions_RW:
			if config.stopCriterea == "convergence":

				print "MAX LEN RW: ", max(len(l) for l in avgBestFitnessExecs_RW)

				for i in range(config.executions):
					avgBestFitnessExecs_RW[i] = avgBestFitnessExecs_RW[i][len(avgBestFitnessExecs_RW[i]) - minLen:]
					avgWorstFitnessExecs_RW[i] = avgWorstFitnessExecs_RW[i][len(avgWorstFitnessExecs_RW[i]) - minLen:]
					avgFitnessVecExecs_RW[i] = avgFitnessVecExecs_RW[i][len(avgFitnessVecExecs_RW[i]) - minLen:]
					avgGenesMatchVecOfVecs_RW[i] = avgGenesMatchVecOfVecs_RW[i][len(avgGenesMatchVecOfVecs_RW[i]) - minLen:]

				print "MAX LEN RW AFTER TRUNCATE: ", max(len(l) for l in avgBestFitnessExecs_RW)

			# From List of Lists to Numpy Array of Lists (len = longest number of generations in executions)
			avgBestFitnessExecsArray_RW = np.array(avgBestFitnessExecs_RW)
			avgWorstFitnessExecsArray_RW = np.array(avgWorstFitnessExecs_RW)
			avgFitnessVecExecsArray_RW = np.array(avgFitnessVecExecs_RW)
			avgGenesMatchVecOfVecsArray_RW = np.array(avgGenesMatchVecOfVecs_RW)
			# (len = number of executions)
			#bestIndividualChmArray_RW = np.array([self.evaluator.decode(i.getChm()) for i in bestIndividualVec_RW])
			bestIndividualMatchesArray_RW = np.array(bestIndividualMatches_RW)

			# Max Matches
			maxMatches_RW = max(bestIndividualMatchesArray_RW)

			finalMeanBest_RW = np.mean(avgBestFitnessExecsArray_RW, axis=0)
			finalMeanWorst_RW = np.mean(avgWorstFitnessExecsArray_RW, axis=0)
			finalMeanAvg_RW = np.mean(avgFitnessVecExecsArray_RW, axis=0)
			finalMeanGeneMatches_RW = np.mean(avgGenesMatchVecOfVecsArray_RW, axis=0)

			finalStdBest_RW = np.std(avgBestFitnessExecsArray_RW, axis=0)
			finalStdWorst_RW = np.std(avgWorstFitnessExecsArray_RW, axis=0)
			finalStdAvg_RW = np.std(avgFitnessVecExecsArray_RW, axis=0)
			finalStdGeneMatches_RW = np.std(avgGenesMatchVecOfVecsArray_RW, axis=0)

			# Precision
			stdGenesBestInd_RW = np.std(bestIndividualChmArray_RW, axis=0)
			precision_RW = np.mean(stdGenesBestInd_RW)

			# Average Matches
			avgMatches_RW = np.mean(bestIndividualMatchesArray_RW)

			# Std Matches
			stdMatches_RW = np.std(bestIndividualMatchesArray_RW)

			# Accuracy
			accuracy_RW = (avgMatches_RW * 100) / len(bestIndividualVec_RW[0].getChm())

			# Save Data
			self.data_to_output.append(["rw", config.populationNumber, config.individualSize, precision_RW,
			 accuracy_RW, solutionFoundGeneration_RW, iterations_RW+1])

			# Save Best From each execution data
			for i in bestIndividualVec_RW:
				self.data_to_output_best_chm_rw.append([i.getChm(), i.getFitness()])

		# Plot Bars for Cooperations and Delations NON STD
		if False:
			n_groups = 10

			ultimateIndex = len(finalMeanCooperation_GA)-1
			penultimateIndex = ultimateIndex - 5


			means_men = list(finalMeanCooperation_GA[0:5])
			means_men += list(finalMeanCooperation_GA[penultimateIndex:ultimateIndex])

			means_women = list(finalMeanDelations_GA[0:5])
			means_women += list(finalMeanDelations_GA[penultimateIndex:ultimateIndex])

			#means_men = np.array(means_men)
			#means_women = np.array(means_women)

			plt = matplot

			fig, ax = matplot.subplots()

			index = np.arange(n_groups)
			bar_width = 0.35

			opacity = 0.4

			rects1 = plt.bar(index, means_men, bar_width,
			                 alpha=opacity,
			                 color='b',
			                 label='Cs')

			rects2 = plt.bar(index + bar_width, means_women, bar_width,
			                 alpha=opacity,
			                 color='r',
			                 label='Ds')

			lenMax = len(finalMeanCooperation_GA)
			indexTruncated = ('1', '2', '3', '4', '5', str(lenMax-4),str(lenMax-3),str(lenMax-2),str(lenMax-1),str(lenMax))

			plt.xlabel('Generation')
			plt.ylabel('Mean')
			plt.title('Average of Cs and Ds in population')
			plt.xticks(index + bar_width / 2, indexTruncated)
			plt.legend()
			plt.tight_layout()
			plt.savefig('charts/%s_barCs_Ds.png' % (configName))
			plt.gcf().clear()

		# Plot Bars for Cooperations and Delations NON STD
		if False:
			n_groups = 10

			ultimateIndex = len(finalMeanPercentCooperations_GA)-1
			penultimateIndex = ultimateIndex - 5


			means_men = list(finalMeanPercentCooperations_GA[0:5])
			means_men += list(finalMeanPercentCooperations_GA[penultimateIndex:ultimateIndex])

			means_women = list(finalMeanPercentDelations_GA[0:5])
			means_women += list(finalMeanPercentDelations_GA[penultimateIndex:ultimateIndex])

			#means_men = np.array(means_men)
			#means_women = np.array(means_women)

			plt = matplot

			fig, ax = matplot.subplots()

			index = np.arange(n_groups)
			bar_width = 0.35

			opacity = 0.4

			rects1 = plt.bar(index, means_men, bar_width,
			                 alpha=opacity,
			                 color='b',
			                 label='Cs')

			rects2 = plt.bar(index + bar_width, means_women, bar_width,
			                 alpha=opacity,
			                 color='r',
			                 label='Ds')

			lenMax = len(finalMeanCooperation_GA)
			indexTruncated = ('1', '2', '3', '4', '5', str(lenMax-4),str(lenMax-3),str(lenMax-2),str(lenMax-1),str(lenMax))

			plt.xlabel('Generation')
			plt.ylabel('Mean in %')
			plt.title('Percentage of Cs and Ds in population')
			plt.xticks(index + bar_width / 2, indexTruncated)
			plt.legend()
			plt.tight_layout()
			plt.savefig('charts/%s_barPercentCs_Ds.png' % (configName))
			plt.gcf().clear()

		# Plot Avg for Cooperations and Delations NON STD
		if False:
			x_axis = np.arange(minLen)

			# Fitness plot
			fit_graph = matplot
			fit_graph.gca().set_color_cycle(['blue', 'red'])
			#axes = fit_graph.gca()
			#axes.set_xlim([0,len(finalMeanAvg)])
			#axes.set_ylim([0,3])

			#fit_graph.plot(x_axis, finalMeanBest)
			#fit_graph.plot(x_axis, finalStdBest)

			#fit_graph.yticks(np.arange(finalMeanBest.min(), finalMeanBest.max(), .001))

			
			fit_graph.plot(x_axis, finalMeanCooperation_GA)
			fit_graph.plot(x_axis, finalMeanDelations_GA)
			
			fit_graph.legend(['C', 'D'], loc='upper center')
			fit_graph.suptitle('Mean - Cs and Ds per Generation')
			fit_graph.xlabel('Generations')
			fit_graph.ylabel('Mean')
			#fit_graph.xticks(np.arange(min(x_axis), max(x_axis)+1, 1))
			#fit_graph.show()
			fit_graph.savefig('charts/%s_lineCs_Ds.png' % (configName))
			fit_graph.gcf().clear()


		# Plot Percent Avg for Cooperations and Delations NON STD
		if False:
			x_axis = np.arange(minLen)

			# Fitness plot
			fit_graph = matplot
			fit_graph.gca().set_color_cycle(['blue', 'red'])
			#axes = fit_graph.gca()
			#axes.set_xlim([0,len(finalMeanAvg)])
			#axes.set_ylim([0,3])

			#fit_graph.plot(x_axis, finalMeanBest)
			#fit_graph.plot(x_axis, finalStdBest)

			#fit_graph.yticks(np.arange(finalMeanBest.min(), finalMeanBest.max(), .001))

			
			fit_graph.plot(x_axis, finalMeanPercentCooperations_GA)
			fit_graph.plot(x_axis, finalMeanPercentDelations_GA)
			
			fit_graph.legend(['C', 'D'], loc='upper center')
			fit_graph.suptitle('Average Percentage - Cs and Ds per Generation')
			fit_graph.xlabel('Generations')
			fit_graph.ylabel('Mean')
			#fit_graph.xticks(np.arange(min(x_axis), max(x_axis)+1, 1))
			#fit_graph.show()
			fit_graph.savefig('charts/%s_linePercentCs_Ds.png' % (configName))
			fit_graph.gcf().clear()

		# NEW STD Plot Avg Mean and Std for AG vs RW with STD
		if True:
			x_axis = np.arange(minLen)

			# Fitness plot
			fit_graph = matplot
			fit_graph.gca().set_color_cycle(['blue'])
			#axes = fit_graph.gca()
			#axes.set_xlim([0,len(finalMeanAvg)])
			#axes.set_ylim([0,3])

			#fit_graph.plot(x_axis, finalMeanBest)
			#fit_graph.plot(x_axis, finalStdBest)

			#fit_graph.yticks(np.arange(finalMeanBest.min(), finalMeanBest.max(), .001))

			#if executions_RW:
			#	fit_graph.errorbar(x_axis, finalMeanAvg_RW, finalStdAvg_RW, fmt='-o')
			
			#if executions_GA:
			#	fit_graph.errorbar(x_axis, finalMeanAvg_GA, finalStdAvg_GA, fmt='-o')


			#fit_graph.errorbar(x_axis, finalMeanAvg_GA, finalStdAvg_GA, fmt='-o')

			error = finalStdAvg_GA
			print "################## ERROR ######################"
			print error

			fit_graph.plot(x_axis, finalMeanAvg_GA, 'k-')
			fit_graph.fill_between(x_axis, finalMeanAvg_GA-error, finalMeanAvg_GA+error, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
			
			fit_graph.legend(['GA'], loc='upper center')
			fit_graph.suptitle('GA - Mean Population Evolution and STD')
			fit_graph.xlabel('Generations')
			fit_graph.ylabel('Fitness')
			#fit_graph.xticks(np.arange(min(x_axis), max(x_axis)+1, 1))
			#fit_graph.show()
			fit_graph.savefig('charts/%s_newMeanFitSTD.png' % (configName))
			fit_graph.gcf().clear()


		# Plot Avg Mean and Std for AG vs RW with STD
		if True:
			x_axis = np.arange(minLen)

			# Fitness plot
			fit_graph = matplot
			fit_graph.gca().set_color_cycle(['blue', 'green'])
			#axes = fit_graph.gca()
			#axes.set_xlim([0,len(finalMeanAvg)])
			#axes.set_ylim([0,3])

			#fit_graph.plot(x_axis, finalMeanBest)
			#fit_graph.plot(x_axis, finalStdBest)

			#fit_graph.yticks(np.arange(finalMeanBest.min(), finalMeanBest.max(), .001))

			if executions_RW:
				fit_graph.errorbar(x_axis, finalMeanAvg_RW, finalStdAvg_RW, fmt='-o')
			
			if executions_GA:
				fit_graph.errorbar(x_axis, finalMeanAvg_GA, finalStdAvg_GA, fmt='-o')
			
			fit_graph.legend(['GA', 'RW'], loc='upper center')
			fit_graph.suptitle('GA - Mean Population Evolution')
			fit_graph.xlabel('Generations')
			fit_graph.ylabel('Fitness')
			#fit_graph.xticks(np.arange(min(x_axis), max(x_axis)+1, 1))
			#fit_graph.show()
			fit_graph.savefig('charts/%s_meanFit.png' % (configName))
			fit_graph.gcf().clear()

		# Plot Avg Mean and Std for AG vs RW NON STD
		if True:
			x_axis = np.arange(minLen)

			# Fitness plot
			fit_graph = matplot
			fit_graph.gca().set_color_cycle(['blue', 'green'])
			#axes = fit_graph.gca()
			#axes.set_xlim([0,len(finalMeanAvg)])
			#axes.set_ylim([0,3])

			#fit_graph.plot(x_axis, finalMeanBest)
			#fit_graph.plot(x_axis, finalStdBest)

			#fit_graph.yticks(np.arange(finalMeanBest.min(), finalMeanBest.max(), .001))

			if executions_RW:
				fit_graph.plot(x_axis, finalMeanAvg_RW)
			
			if executions_GA:
				fit_graph.plot(x_axis, finalMeanAvg_GA)
			
			fit_graph.legend(['GA', 'RW'], loc='upper center')
			fit_graph.suptitle('GA - Mean Population Evolution NO STD')
			fit_graph.xlabel('Generations')
			fit_graph.ylabel('Fitness')
			#fit_graph.xticks(np.arange(min(x_axis), max(x_axis)+1, 1))
			#fit_graph.show()
			fit_graph.savefig('charts/%s_meanFitNOSTD.png' % (configName))
			fit_graph.gcf().clear()

		# Plot Best, Avg and Worst Mean
		if True:
			x_axis = np.arange(minLen)

			# Fitness plot
			fit_graph = matplot
			fit_graph.gca().set_color_cycle(['green', 'blue', 'red'])
			#axes = fit_graph.gca()
			#axes.set_xlim([0,len(finalMeanAvg)])
			#axes.set_ylim([0,3])

			fit_graph.plot(x_axis, finalMeanBest_GA)
			fit_graph.plot(x_axis, finalMeanAvg_GA)
			fit_graph.plot(x_axis, finalMeanWorst_GA)

			#fit_graph.yticks(np.arange(finalMeanBest.min(), finalMeanBest.max(), .001))

			#if executions_RW:
			#	fit_graph.errorbar(x_axis, finalMeanBest_RW, finalStdBest_RW, fmt='-o')
			
			#if executions_GA:
			#	fit_graph.errorbar(x_axis, finalMeanBest_GA, finalStdBest_GA, fmt='-o')
			
			fit_graph.legend(['Best', 'Avg', 'Worst'], loc='upper center')
			fit_graph.suptitle('GA - Best, Avg and Worst Individual Evolution')
			fit_graph.xlabel('Generations')
			fit_graph.ylabel('Fitness')
			#fit_graph.xticks(np.arange(min(x_axis), max(x_axis)+1, 1))
			#fit_graph.show()
			fit_graph.savefig('charts/%s_BestAvgWorst.png' % (configName))
			fit_graph.gcf().clear()


		# Plot Best, Mean and Worst Fitness Avg
		if False:
			x_axis = np.arange(minLen)

			# Fitness plot
			fit_graph = matplot
			fit_graph.gca().set_color_cycle(['blue', 'green'])
			#axes = fit_graph.gca()
			#axes.set_xlim([0,len(finalMeanAvg)])
			#axes.set_ylim([0,3])


			if executions_RW:
				fit_graph.plot(x_axis, finalMeanBest_RW)
			
			if executions_GA:
				fit_graph.plot(x_axis, finalMeanBest_GA)
			
			fit_graph.legend(['GA', 'RW'], loc='upper_left')
			fit_graph.suptitle('GA - Best Individual Evolution NO STD')
			fit_graph.xlabel('Generations')
			fit_graph.ylabel('Fitness')
			#fit_graph.show()
			fit_graph.savefig('charts/ga_NOSTD_best_fitness_pop%s_word%s.png' % (config.populationNumber, config.individualSize))
			fit_graph.gcf().clear()

		# Plot Only RW 
		if False:
			x_axis = np.arange(minLen)
			fit_graph = matplot
			fit_graph.gca().set_color_cycle(['green'])
			#axes = fit_graph.gca()
			#axes.set_xlim([0,len(finalMeanAvg)])
			#axes.set_ylim([0,3])

			#fit_graph.plot(x_axis, finalMeanBest)
			fit_graph.errorbar(x_axis, finalMeanBest_RW, finalStdBest_RW, fmt='-o')
			#fit_graph.plot(x_axis, finalMeanWorst)
			
			fit_graph.legend(['RW'], loc='upper_left')
			fit_graph.suptitle('RW - Best Individual Evolution')
			fit_graph.xlabel('Generations')
			fit_graph.ylabel('Fitness')
			#fit_graph.show()
			fit_graph.savefig('charts/onlyRW_best_fitness_pop%s_word%s.png' % (config.populationNumber, config.individualSize))
			fit_graph.gcf().clear()

		# Plot Only GA
		if False:
			x_axis = np.arange(minLen)
			fit_graph = matplot
			fit_graph.gca().set_color_cycle(['blue'])
			#axes = fit_graph.gca()
			#axes.set_xlim([0,len(finalMeanAvg)])
			#axes.set_ylim([0,3])

			#fit_graph.plot(x_axis, finalMeanBest)
			fit_graph.errorbar(x_axis, finalMeanBest_GA, finalStdBest_GA, fmt='-o')
			#fit_graph.plot(x_axis, finalMeanWorst)
			
			fit_graph.legend(['GA'], loc='upper_left')
			fit_graph.suptitle('GA - Best Individual Evolution')
			fit_graph.xlabel('Generations')
			fit_graph.ylabel('Fitness')
			#fit_graph.show()
			fit_graph.savefig('charts/onlyGA_best_fitness_pop%s_word%s.png' % (config.populationNumber, config.individualSize))
			fit_graph.gcf().clear()


		# Plot Genes Matches 
		if False:
			fit_graph = matplot
			fit_graph.gca().set_color_cycle(['blue'])
			#axes = fit_graph.gca()
			#axes.set_xlim([0,len(finalMeanAvg)])
			#axes.set_ylim([0,3])

			#fit_graph.plot(x_axis, finalMeanBest)
			fit_graph.plot(x_axis, finalMeanGeneMatches)
			#fit_graph.plot(x_axis, finalMeanWorst)
			
			fit_graph.legend(['Mean Genes Matches'], loc='upper_left')
			fit_graph.suptitle('Average Genes Matches per Executions')
			fit_graph.xlabel('Generations')
			fit_graph.ylabel('Genes Matches')
			fit_graph.show()



	def computeOverallStats(self, executions):
		avgBestFitnessExecs = []
		avgWorstFitnessExecs = []
		avgFitnessVecExecs = []
		avgGenesMatchVecOfVecs = []

		for exe in executions:
			avgBestFitnessExecs.append(exe.stats.getBestFitnessVec())
			avgWorstFitnessExecs.append(exe.stats.getWorstFitnessVec())
			avgFitnessVecExecs.append(exe.stats.getAvgFitnessVec())
			avgGenesMatchVecOfVecs.append(exe.stats.getAvgGenesMatchesVec())

		totalIterations = len(avgFitnessVecExecs[0])
		totalExecutions = len(executions)

		avgBestFitnessExecsArray = np.array(avgBestFitnessExecs)
		avgWorstFitnessExecsArray = np.array(avgWorstFitnessExecs)
		avgFitnessVecExecsArray = np.array(avgFitnessVecExecs)
		avgGenesMatchVecOfVecsArray = np.array(avgGenesMatchVecOfVecs)

		finalMeanBest = np.mean(avgBestFitnessExecsArray, axis=0)
		finalMeanWorst = np.mean(avgWorstFitnessExecsArray, axis=0)
		finalMeanAvg = np.mean(avgFitnessVecExecsArray, axis=0)
		finalMeanGeneMatches = np.mean(avgGenesMatchVecOfVecsArray, axis=0)

		finalStdBest = np.std(avgBestFitnessExecsArray, axis=0)
		finalStdWorst = np.std(avgWorstFitnessExecsArray, axis=0)
		finalStdAvg = np.std(avgFitnessVecExecsArray, axis=0)
		finalStdGeneMatches = np.std(avgGenesMatchVecOfVecsArray, axis=0)

		# Plot Best Mean and Std for AG vs RW
		if True:
			x_axis = np.arange(len(avgBestFitnessExecsArray_GA))

			# Fitness plot
			fit_graph = matplot
			fit_graph.gca().set_color_cycle(['green', 'blue'])
			#axes = fit_graph.gca()
			#axes.set_xlim([0,len(finalMeanAvg)])
			#axes.set_ylim([0,3])

			#fit_graph.plot(x_axis, finalMeanBest)
			#fit_graph.plot(x_axis, finalStdBest)

			#fit_graph.yticks(np.arange(finalMeanBest.min(), finalMeanBest.max(), .001))

			fit_graph.errorbar(x_axis, finalMeanBest, finalStdBest, fmt='-o')
			fit_graph.errorbar(x_axis, finalMeanAvg, finalStdAvg, fmt='-o')
			
			fit_graph.legend(['Best', 'Avg'], loc='top')
			fit_graph.suptitle('Average and Std Fitness per Executions')
			fit_graph.xlabel('Generations')
			fit_graph.ylabel('Fitness')
			fit_graph.show()


		# Plot Best, Mean and Worst Fitness Avg
		if False:
			x_axis = np.arange(len(finalMeanAvg))

			# Fitness plot
			fit_graph = matplot
			fit_graph.gca().set_color_cycle(['green', 'blue', 'red'])
			#axes = fit_graph.gca()
			#axes.set_xlim([0,len(finalMeanAvg)])
			#axes.set_ylim([0,3])

			fit_graph.plot(x_axis, finalMeanBest)
			fit_graph.plot(x_axis, finalMeanAvg)
			fit_graph.plot(x_axis, finalMeanWorst)
			
			fit_graph.legend(['Best', 'Mean', 'Worst'], loc='upper_left')
			fit_graph.suptitle('Average Fitness per Executions')
			fit_graph.xlabel('Generations')
			fit_graph.ylabel('Fitness')
			fit_graph.show()


		# Plot Genes Matches 
		if False:
			fit_graph = matplot
			fit_graph.gca().set_color_cycle(['blue'])
			#axes = fit_graph.gca()
			#axes.set_xlim([0,len(finalMeanAvg)])
			#axes.set_ylim([0,3])

			#fit_graph.plot(x_axis, finalMeanBest)
			fit_graph.plot(x_axis, finalMeanGeneMatches)
			#fit_graph.plot(x_axis, finalMeanWorst)
			
			fit_graph.legend(['Mean Genes Matches'], loc='upper_left')
			fit_graph.suptitle('Average Genes Matches per Executions')
			fit_graph.xlabel('Generations')
			fit_graph.ylabel('Genes Matches')
			fit_graph.show()

