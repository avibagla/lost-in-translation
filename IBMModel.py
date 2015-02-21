class IBM_Model_1:
	"""Class which trains IBM Model 1 and allows for testing"""

	def __init__(self):
		self.englishCorpus = loadList('./es-en/train/europarl-v7.es-en.en')
		self.spanishCorpus = loadList('./es-en/train/europarl-v7.es-en.es')
		self.englishVocabulary = set()
		self.spanishVocabulary = set()
		for i, sentence in enumerate(self.englishCorpus):
			self.englishVocabulary |= set(self.englishCorpus[i].split())
			self.spanishVocabulary |= set(self.spanishCorpus[i].split())


	def train(self):
		pass

	def EMOneInstance(expectationList, primaryLanguageVocab, secondaryLanguageVocab):
		probabalityGrid = {}
		alignmentProbabilities = Estep([{"green house", "casa verde"}, {"the house", "la casa"}], probabalityGrid)
		Mstep([alignmentProbabilities, probabalityGrid])
		return probabalityGrid

	def Estep(trainingPhrases, translationProbGrid):
		"""Runs the Expectation Step of the IBM Model 1 algorithm"""
		alignmentProbabilities = []
		return alignmentProbabilities

	def Mstep(alignmentProbabilities, trandlationProbGrid):
		"""Runs the Maximization step of the IBM Model 1 algorithm"""
		pass



def loadList(file_name):
    """Loads text files as lists of lines. Used in evaluation."""
    """Taken from pa5"""
    with open(file_name) as f:
        l = [line.strip() for line in f]
    f.close()
    return l



def main():
	IBM_Model = IBM_Model_1()

main()