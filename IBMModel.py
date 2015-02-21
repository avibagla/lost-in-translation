class IBM_Model_1:
	'''Class which trains IBM Model 1 and allows for testing'''

	def __init__(self):
		self.englishTrainList = loadList('./es-en/train/europarl-v7.es-en.en')
		self.spanishTrainList = loadList('./es-en/train/europarl-v7.es-en.es')
		print len(self.englishTrainList)

	def train(self):
		pass

	def EMOneInstance(expectationList, primaryLanguageVocab, secondaryLanguageVocab):
		probabalityGrid = {}
		alignmentProbabilities = Estep([{"green house", "casa verde"}, {"the house", "la casa"}], probabalityGrid)
		Mstep([alignmentProbabilities, probabalityGrid])
		return probabalityGrid

	def Estep(trainingPhrases, translationProbGrid):
		alignmentProbabilities = []
		return alignmentProbabilities

	def Mstep(alignmentProbabilities, trandlationProbGrid):
		pass



def loadList(file_name):
    """Loads text files as lists of lines. Used in evaluation."""
    """Taken from pa5"""
    with open(file_name) as f:
        l = [line.strip() for line in f]
    return l