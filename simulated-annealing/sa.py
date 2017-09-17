
import math
from solutions import BinarySolution

class SimulatedAnnealing(object):
	"""docstring for SimulatedAnnealing"""
	def __init__(self):
		self.currentSolution = BinarySolution(966, None)
		self.T = 1000
		self.finalT = 0
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

		return neighbour

	def selectNewSolution(self, solution):
		scoreCurrent = evaluateSolution(self.currentSolution)

		

		

	def evaluateSolution(self):
		pass

	def cool(self):
		pass



		