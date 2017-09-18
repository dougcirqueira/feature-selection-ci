
import math
import random
import numpy as np

from solutions import BinarySolution
from evaluators import SVMEvaluator

class SimulatedAnnealing(object):
	"""docstring for SimulatedAnnealing"""
	def __init__(self):
		self.currentSolution = BinarySolution(966, None)
		self.T = 10
		self.finalT = 0.1
		self.TR = 2
		self.coolingFactor = 0.3
		self.h = 2
		self.lenSolution = len(self.currentSolution.getChm())
		self.evaluator = SVMEvaluator()

	def acceptanceFunction(self, scoreNeighbour, scoreCurrent):
		exponent = (scoreNeighbour - scoreCurrent) / self.T
		p = pow(math.e, exponent)
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
		solution = self.currentSolution

		# While Temperature is high enough
		while self.T > self.finalT:
			# Loop for number of selections with same temperature
			for q in xrange(self.TR):
				# Look for new neighbour solution
				neighbour = self.getNeighbour(solution.getChm())

				# Select new solution based on current one and neighbour
				solution = self.selectNewSolution(solution, neighbour)

			# Cool system
			self.T = self.cool(self.T, self.coolingFactor)

			print "---------------------------------------------------------------------------"
			print "Temperature: %f" % self.T
			print "Current Score: %f" % solution.getFitness()
			print "---------------------------------------------------------------------------"

		self.currentSolution = solution



def main():
	sa = SimulatedAnnealing()
	sa.run()


if __name__ == '__main__':
	main()