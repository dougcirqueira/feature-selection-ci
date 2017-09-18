
import math
import random
import numpy as np
import csv
import matplotlib.pyplot as matplot

from solutions import BinarySolution
from evaluators import SVMEvaluator
from collections import Counter

class SimulatedAnnealing(object):
	"""docstring for SimulatedAnnealing"""
	def __init__(self):
		self.currentSolution = BinarySolution(966, None)
		self.T = 1000
		self.maxT = self.T
		self.finalT = 0.1
		self.TR = 2
		self.coolingFactor = 0.95
		self.h = 10	
		self.lenSolution = len(self.currentSolution.getChm())
		self.evaluator = SVMEvaluator()
		self.population = []
		self.temperatureRecord = []
		self.solutionByTemp = []
		self.totalSelected = []
		self.totalRemoved = []
		self.iterationsConvergence = 50

	def acceptanceFunction(self, scoreNeighbour, scoreCurrent):
		scoreNeighbour *= self.maxT
		scoreCurrent *= self.maxT
		exponent = (scoreNeighbour - scoreCurrent) / self.T
		p = pow(math.e, exponent)
		print "P %f", p
		return p

	def getNeighbour(self, solution):
		neighbour = list(solution)
		indexes = range(0, self.lenSolution)
		positions = np.random.choice(indexes, size=self.h, replace=False)
		for pos in positions:
			if neighbour[pos] == 1:
				neighbour[pos] = 0
			else:
				neighbour[pos] = 1

		return  BinarySolution(self.lenSolution, neighbour)

	def selectNewSolution(self, solution, neighbour):

		solution.setFitness(self.evaluateSolution(solution.getChm()))
		scoreCurrent = solution.getFitness()
		neighbour.setFitness(self.evaluateSolution(neighbour.getChm()))
		scoreNeighbour = neighbour.getFitness()

		if scoreCurrent < scoreNeighbour:
			return neighbour

		elif random.uniform(0,1) < self.acceptanceFunction(scoreNeighbour, scoreCurrent):
			return neighbour

		else:
			return solution


	def evaluateSolution(self, solution):
		score = self.evaluator.evaluateWithStats(solution)
		return score

	def cool(self, T, coolingFactor):
		return T * coolingFactor

	def run(self):

		countSurpassesNeuralNet = 0
		iteration = 0
		countGeneSameBestFit = 0
		lastBestFitness = 0

		# Save Best From each execution data
		self.data_to_output_best_chm_ga = []
		self.data_to_output_best_chm_ga.append(["generation", "fitness", "features",
			"precPosFinal","precNegFinal", "precAvg",  "recallPosFinal", 
			"recallNegFinal", "recallAvg", "f1PosFinal", "f1NegFinal", "macrof1Final", "featuresSelected",
			"featuresRemoved", "chromossome"])

		solution = self.currentSolution

		self.population.append(solution)


		# While Temperature is high enough
		while self.T > self.finalT:
			# Loop for number of selections with same temperature
			for q in xrange(self.TR):
				# Look for new neighbour solution
				neighbour = self.getNeighbour(solution.getChm())

				self.population.append(neighbour)

				# Select new solution based on current one and neighbour
				solution = self.selectNewSolution(solution, neighbour)

				if True:
				#if solution.getFitness() > 0.8073:
					i = solution
					chromossome = i.getChm()

					selected, removed = self.evaluator.getFeatSelectedRemoved(chromossome)

					self.totalSelected += selected
					self.totalRemoved += removed

					features = Counter(chromossome)
					self.data_to_output_best_chm_ga.append([
						iteration,
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
						selected,
						removed,
						chromossome
					])

					with open("../results_sa/%s_"%("T-"+str(self.maxT)+"_"+"h-"+str(self.h)) + "_neuralSurpassesBestsGA.csv", "a") as output:
						writer = csv.writer(output)
						writer.writerows(self.data_to_output_best_chm_ga)

					self.data_to_output_best_chm_ga = []				

					# Save All Population from Best Execution
					self.data_to_output_allPop = []
					self.data_to_output_allPop.append(["generation", "fitness", "features",
						"precPosFinal","precNegFinal", "precAvg",  "recallPosFinal", 
						"recallNegFinal", "recallAvg", "f1PosFinal", "f1NegFinal", "macrof1Final", "featuresSelected",
						"featuresRemoved", "chromossome"])

					populationToExport = self.population

					for i in populationToExport:
						chromossome = i.getChm()
						selected, removed = self.evaluator.getFeatSelectedRemoved(chromossome)
						features = Counter(chromossome)
						self.data_to_output_allPop.append([
							iteration,
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
							selected,
							removed,
							chromossome
						])

					with open("../results_sa/%s_"%("T-"+str(self.maxT)+"_"+"h-"+str(self.h)) + "neuralSurpassesAllPopulationGA.csv", "w") as output:
						writer = csv.writer(output)
						writer.writerows(self.data_to_output_allPop)

					countSurpassesNeuralNet += 1

					iteration += 1

			# Cool system
			self.temperatureRecord.append(self.T)
			self.T = self.cool(self.T, self.coolingFactor)
			
			self.solutionByTemp.append(solution.getFitness())

			print "---------------------------------------------------------------------------"
			print "Temperature: %f" % self.T
			print "Current Score: %f" % solution.getFitness()
			print "---------------------------------------------------------------------------"

			currBestFitness = solution.getFitness()

			countGeneSameBestFit += 1

			if currBestFitness > lastBestFitness:
				print "Zerando"
				countGeneSameBestFit = 0

			if countGeneSameBestFit == self.iterationsConvergence:
				print "CONVERGENCE"
				print "INITIAL GENERATION (%d) and END (%d)" % (iteration-self.iterationsConvergence, iteration)
				break
				return

			lastBestFitness = solution.getFitness()


		self.currentSolution = solution

		#self.plotCharts(self.population)


		# Save Most Common From each execution data
		self.data_to_output_most_common_selected = []
		self.data_to_output_most_common_selected.append(["mostCommon"])

		self.data_to_output_most_common_removed = []
		self.data_to_output_most_common_removed.append(["mostCommon"])

		countSelected = Counter(self.totalSelected)
		countRemoved = Counter(self.totalRemoved)

		self.data_to_output_most_common_selected.append(countSelected.most_common())
		self.data_to_output_most_common_removed.append(countRemoved.most_common())

		with open("../results_sa/%s_"%("T-"+str(self.maxT)+"_"+"h-"+str(self.h)) + "_selectedMostCommon.csv", "a") as output:
			writer = csv.writer(output)
			writer.writerows(self.data_to_output_most_common_selected)

		with open("../results_sa/%s_"%("T-"+str(self.maxT)+"_"+"h-"+str(self.h)) + "_removedMostCommon.csv", "a") as output:
			writer = csv.writer(output)
			writer.writerows(self.data_to_output_most_common_removed)


def plotCharts(population, solutionByTemp, temperatureRecord):

		scoresPopulation = []

		for pop in population:
			scores = [i.getFitness() for i in pop]			
			scoresPopulation.append(scores)			

		population = scoresPopulation
		
		minLenPop = min(len(l) for l in population)
		for i in range(len(population)):
			population[i] = population[i][len(population[i]) - minLenPop:]
		
		minLenSolByTemp = min(len(l) for l in solutionByTemp)
		for i in range(len(solutionByTemp)):
			solutionByTemp[i] = solutionByTemp[i][len(solutionByTemp[i]) - minLenSolByTemp:]

		temperatureRecord = temperatureRecord[len(temperatureRecord) - minLenSolByTemp:]


		avgScores = np.array(population)
		avgSolutionByTemp = np.array(solutionByTemp)

		avgScoresMean = np.mean(avgScores, axis=0)
		avgScoresSTD = np.std(avgScores, axis=0)

		avgSolutionByTempMean = np.mean(avgSolutionByTemp, axis=0)
		avgSolutionByTempSTD = np.std(avgSolutionByTemp, axis=0)


		# Score x Iterations
		x_axis = np.arange(minLenPop)

		# Fitness plot
		fit_graph = matplot
		fit_graph.gca().set_color_cycle(['green'])

		error = avgScoresSTD
		print "################## ERROR ######################"
		#print error

		fit_graph.plot(x_axis, avgScoresMean, 'k-')
		fit_graph.fill_between(x_axis, avgScoresMean-error, avgScoresMean+error, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
		
		fit_graph.legend(['SA'], loc='upper center')
		fit_graph.suptitle('SA - Score of Solutions by Iterations')
		fit_graph.xlabel('Iteration')
		fit_graph.ylabel('Score')
		#fit_graph.xticks(np.arange(min(x_axis), max(x_axis)+1, 1))
		#fit_graph.show()
		fit_graph.savefig('../results_sa/charts/%s_scoreXiteration.png' % ("T-"+str(100)+"_"+"h-"+str(2)))
		fit_graph.gcf().clear()


		# Score x Temperature
		#x_axis = np.arange(minLenSolByTemp)
		x_axis = np.array(temperatureRecord)
		#x_axis = np.flipud(x_axis)

		#avgSolutionByTempMean = np.flipud(avgSolutionByTempMean)

		# Fitness plot
		fit_graph = matplot
		fit_graph.gca().set_color_cycle(['green'])

		#error = np.flipud(avgSolutionByTempSTD)
		error = avgSolutionByTempSTD
		print "################## ERROR ######################"
		#print error

		fit_graph.plot(x_axis, avgSolutionByTempMean, 'k-')
		fit_graph.fill_between(x_axis, avgSolutionByTempMean-error, avgSolutionByTempMean+error, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
		
		fit_graph.legend(['SA'], loc='upper center')
		fit_graph.suptitle('SA - Score of Solutions by Temperature')
		fit_graph.xlabel('Temperature')
		fit_graph.ylabel('Score')
		#fit_graph.xticks(np.arange(min(x_axis), max(x_axis)+1, 1))
		#fit_graph.show()
		fit_graph.savefig('../results_sa/charts/%s_scoreXtemperature.png' % ("T-"+str(100)+"_"+"h-"+str(2)))
		fit_graph.gcf().clear()




		"""
		# Score x Iterations
		x_axis = np.arange(len(population))

		scores = [i.getFitness() for i in population]

		# Fitness plot
		fit_graph = matplot
		fit_graph.gca().set_color_cycle(['green'])
		#axes = fit_graph.gca()
		#axes.set_xlim([0,len(finalMeanAvg)])
		#axes.set_ylim([0,3])

		#fit_graph.plot(x_axis, finalMeanBest)
		#fit_graph.plot(x_axis, finalStdBest)

		#fit_graph.yticks(np.arange(finalMeanBest.min(), finalMeanBest.max(), .001))

		fit_graph.plot(x_axis, scores)
		
		fit_graph.legend(['SA'], loc='upper center')
		fit_graph.suptitle('SA - Score of Solutions by Iterations')
		fit_graph.xlabel('Interation')
		fit_graph.ylabel('Score')
		#fit_graph.xticks(np.arange(min(x_axis), max(x_axis)+1, 1))
		#fit_graph.show()
		fit_graph.savefig('../results_sa/charts/%s_scoreXiteration.png' % ("T-"+str(self.maxT)+"_"+"h-"+str(self.h)))
		fit_graph.gcf().clear()


		# Score x Temperature
		x_axis = self.temperatureRecord

		scores = self.solutionByTemp

		# Fitness plot
		fit_graph = matplot
		fit_graph.gca().set_color_cycle(['green'])
		#axes = fit_graph.gca()
		#axes.set_xlim([0,len(finalMeanAvg)])
		#axes.set_ylim([0,3])

		#fit_graph.plot(x_axis, finalMeanBest)
		#fit_graph.plot(x_axis, finalStdBest)

		#fit_graph.yticks(np.arange(finalMeanBest.min(), finalMeanBest.max(), .001))

		fit_graph.plot(x_axis, scores)
		
		fit_graph.legend(['SA'], loc='upper center')
		fit_graph.suptitle('SA - Score of Population by Temperature')
		fit_graph.xlabel('Temperature')
		fit_graph.ylabel('Score')
		#fit_graph.xticks(np.arange(min(x_axis), max(x_axis)+1, 1))
		#fit_graph.show()
		fit_graph.savefig('../results_sa/charts/%s_scoreXtemperature.png' % ("T-"+str(self.maxT)+"_"+"h-"+str(self.h)))
		fit_graph.gcf().clear()

		"""

def main():

	popVecofVecs = []
	solutionByTempVecofVecs = []

	for exe in range(0,5):
		sa = SimulatedAnnealing()
		sa.run()
		popVecofVecs.append(sa.population)
		solutionByTempVecofVecs.append(sa.solutionByTemp)

	plotCharts(popVecofVecs, solutionByTempVecofVecs, sa.temperatureRecord)


if __name__ == '__main__':
	main()