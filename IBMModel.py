from collections import defaultdict
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
		#alignmentProbabilities = Estep([{"green house", "casa verde"}, {"the house", "la casa"}], probabalityGrid)
		spanishCountDict = {}
		countInv = 1./(len(self.spanishVocabulary)*len(self.englishVocabulary))
		for word in self.spanishVocabulary:
			spanishCountDict[word] = countInv
		self.translate = {}
		for word in self.englishVocabulary:
			self.translate[word] = spanishCountDict.copy()


	def EMOneInstance(self, expectationList, primaryLanguageVocab, secondaryLanguageVocab):

		return Mstep(alignmentProbabilities, probabalityGrid)

	def Estep(self, trainingPhrases, translationProbGrid):
		"""Runs the Expectation Step of the IBM Model 1 algorithm"""
		for englishSentence, foreignSentence in trainingPhrases:
			#We first need to compute P(a, f |e) by multiplying all the t probabilities, following
			pass
		return alignmentProbabilities

	def Mstep(self, alignmentProbabilities, translationProbGrid):
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
	IBM_Model.train()
	print "done"

main()