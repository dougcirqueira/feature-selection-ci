

class BinarySolution(object):
	"""docstring for BinarySolution"""
	def __init__(self, nGenes, chromossome=None):
		self.chm = [random.randint(0,1) for i in range(nGenes)] if chromossome == None else chromossome
		self.fitness = 0
		self.genesMatches = 0
		self.precisions_posFinal = 0
		self.precisions_negFinal = 0
		self.recalls_posFinal = 0
		self.recalls_negFinal = 0
		self.f1_scores_posFinal = 0
		self.f1_scores_negFinal = 0
		self.macro_f1Final = 0
		
	def setChm(self, chm):
		self.chm = chm

	def getChm(self):
		return self.chm

	def setFitness(self, fitness):
		self.fitness = fitness[0]
		self.precisions_posFinal = fitness[1]
		self.precisions_negFinal = fitness[2]
		self.recalls_posFinal = fitness[3]
		self.recalls_negFinal = fitness[4]
		self.f1_scores_posFinal = fitness[5]
		self.f1_scores_negFinal = fitness[6]
		self.macro_f1Final = fitness[7]

	
	def getFitness(self):
		return self.fitness

	def setGenesMatches(self, genesMatches):
		self.genesMatches = genesMatches

	def getGenesMatches(self):
		return self.genesMatches