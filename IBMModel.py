class IBM_Model_1:
	'''Class which trains IBM Model 1 and allows for testing'''

	def __init__(self):
		pass

	def train(self):
		pass

	def EMOneInstance(expectationList, primaryLanguageVocab, secondaryLanguageVocab):
		probabalityGrid = [][]
		alignmentProbabilities = Estep([{"green house", "casa verde"}, {"the house", "la casa"}], probabalityGrid)
		Mstep([alignmentProbabilities, probabalityGrid])
		return probabalityGrid

	def Estep(trainingPhrases, translationProbGrid):
		alignmentProbabilities = []
		return alignmentProbabilities

	def Mstep(alignmentProbabilities, trandlationProbGrid):
		pass



def loadList()