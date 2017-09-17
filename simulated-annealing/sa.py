
import math
import random

from solutions import BinarySolution

class SimulatedAnnealing(object):
	"""docstring for SimulatedAnnealing"""
	def __init__(self):
		self.currentSolution = BinarySolution(966, None)
		self.T = 100
		self.finalT = 0
		self.TR = 5
		self.coolingFactor = 0.3
		self.h = 100
		


		self.lenSolution = len(self.currentSolution.getChm())

	def acceptanceFunction(self, scoreNeighbour, scoreCurrent):
		p = pow(math.e, (-scoreNeighbour - scoreCurrent) / self.T)
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

		scoreCurrent = evaluateSolution(solution.getChm())
		scoreNeighbour = evaluateSolution(neighbour.getChm())

		if scoreCurrent < scoreNeighbour:
			return neighbour

		elif random.uniform(0,1) < self.acceptanceFunction(scoreNeighbour, scoreCurrent):
			return neighbour

		else:
			return solution


	def evaluateSolution(self, solution):
		pass

	def cool(self, T, coolingFactor):
		return T * coolingFactor

	def run(self):
		pass

		