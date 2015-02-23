import time
from collections import defaultdict
englishCorpusFile = './es-en/train/europarl-v7.es-en.en' #'../es-en/train/small.en' 
spanishCorpusFile = './es-en/train/europarl-v7.es-en.es' # '../es-en/train/small.es'


class IBM_Model_1:
	"""Class which trains IBM Model 1 and allows for testing"""


	def __init__(self):
		self.englishCorpus = loadList(englishCorpusFile)
		self.spanishCorpus = loadList(spanishCorpusFile)
		self.englishVocabulary = set()
		self.spanishVocabulary = set()
		self.null = "<<NULL>>"
		for i, sentence in enumerate(self.englishCorpus):
			self.englishVocabulary |= set(self.englishCorpus[i].split())
			self.spanishVocabulary |= set(self.spanishCorpus[i].split())
		self.englishVocabulary.add(self.null)


	def train(self):
		self.englishToIndex = {}
		self.spanishToIndex = {}
		countInv = 1./(len(self.spanishVocabulary))
		self.translate = [[countInv]*len(self.spanishVocabulary) for i in range(len(self.englishVocabulary))]
		i = 0
		for word in self.englishVocabulary:
			self.englishToIndex[word] = i
			i+=1
		i = 0
		for word in self.spanishVocabulary:
			self.spanishToIndex[word] = i
			i+=1
		print "table is built"
		self.Estep(zip(self.englishCorpus, self.spanishCorpus))
		print "E Step completed"
		self.Mstep()
		print "M Step completed"
		# for f in self.spanishVocabulary:
		# 	for e in self.englishVocabulary:
		# 		print e,f,self.translate[self.englishToIndex[e]][self.spanishToIndex[f]]



	def EMOneInstance(self, expectationList, primaryLanguageVocab, secondaryLanguageVocab):

		return Mstep(alignmentProbabilities, probabalityGrid)

	def Estep(self, trainingPhrases):
		"""Runs the Expectation Step of the IBM Model 1 algorithm"""
		newTranslate = [[0]*len(self.spanishVocabulary) for i in range(len(self.englishVocabulary))]
		for englishSentence, foreignSentence in trainingPhrases:
			englishSentence = englishSentence.split()
			englishSentence.append(self.null)
			foreignSentence = foreignSentence.split()
			#We first need to compute P(a, f |e) by multiplying all the t probabilities, following
			translationProbability = self.findSentenceTranslationProb(englishSentence, foreignSentence)
			for i in range(len(foreignSentence)):
				for e in englishSentence:
					f = foreignSentence[i]
					newForeignSentence = list(foreignSentence)
					newForeignSentence.pop(i)
					transProbWithFixedEF = self.findSentenceTranslationProb(englishSentence, newForeignSentence)
					t_ef = self.translate[self.englishToIndex[e]][self.spanishToIndex[f]] 
					newTranslate[self.englishToIndex[e]][self.spanishToIndex[f]] += (t_ef*transProbWithFixedEF/translationProbability if translationProbability > 0 else 0)
					#print e,f, t_ef*transProbWithFixedEF/translationProbability
		self.translate = newTranslate
		#print self.translate

	def Mstep(self):
		"""Runs the Maximization step of the IBM Model 1 algorithm"""
		for i in range(len(self.englishVocabulary)):
			rowTotal = 0
			for j in range(len(self.spanishVocabulary)):
				rowTotal += self.translate[i][j]
			for j in range(len(self.spanishVocabulary)):
				self.translate[i][j]/= float(rowTotal)

	def findSentenceTranslationProb(self, englishSentence, foreignSentence):
		normalizationFactor = 1.
		for f in foreignSentence:
			insideSum = 0
			for e in englishSentence:
				insideSum += self.translate[self.englishToIndex[e]][self.spanishToIndex[f]] 
			normalizationFactor *= insideSum
		return normalizationFactor

	def predict(self, inputSentence):
		"""Takes in a foreign sentence and uses predictions to determine the highest liklihood sentence with our MT"""
		pass


def loadList(file_name):
    """Loads text files as lists of lines."""
    """Taken from pa5"""
    with open(file_name) as f:
        l = [line.strip() for line in f]
    f.close()
    return l



def main():
	start = time.clock()
	IBM_Model = IBM_Model_1()
	IBM_Model.train() 
	print time.clock() - start
main()