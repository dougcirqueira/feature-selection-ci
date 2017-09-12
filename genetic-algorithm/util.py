
import numpy as np

class PrisonerUtil(object):
	def __init__(self, bonusMin, bonusMax, cooperationSize):
		self.oldMin = 0.5
		self.oldMax = 0.0
		self.bonusMin = bonusMin
		self.bonusMax = bonusMax
		self.cooperationSize = cooperationSize


	def calcBonus(self, individual):
		# Bonus Calculation
		chromossome = individual.getChm()
		len_chm = len(chromossome)

		# Retrieve 
		chains = []
		curr_chain = []
		for i in range(len_chm):
			gene = chromossome[i]
			if gene < 0.5:
				curr_chain.append(gene)

				if i == len_chm-1:
					if len(curr_chain) >= self.cooperationSize:	
						chains.append(curr_chain)
			else:
				if len(curr_chain) >= self.cooperationSize: 
					chains.append(curr_chain)

				curr_chain = []
		
		fitnessWithBonus = 0.

		meanChainsSum = 0.
		for chain in chains:
			meanChainsSum += np.mean(chain) / len(chain)

		if meanChainsSum > 0:
			meanGenes = meanChainsSum / len(chains)

			bonus = (((meanGenes - self.oldMin) * (self.bonusMax - self.bonusMin)) / (self.oldMax-self.oldMin)) + self.bonusMin

			fitness = individual.getFitness()

			fitnessWithBonus = fitness + ((bonus/100) * (1 - fitness))
			#fitnessWithBonus = fitness + ((bonus/100) * (fitness))

		return fitnessWithBonus
