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



	def Estep(self, trainingPhrases):
		"""
			Runs the Expectation Step of the IBM Model 1 algorithm
			i.e. 
			for each sentence pair
				for each e in E and f in F
					Add the total contribution of the sentences E,F to the word pair: tranlate(e,f)
			"""
		newTranslate = [[0]*len(self.spanishVocabulary) for i in range(len(self.englishVocabulary))]
		self.totalF = [0 for i in range(len(self.spanishVocabulary))]

		for englishSentence, foreignSentence in trainingPhrases:
			# Tokenize sentences and add NULL to englishSentence
			englishSentence = englishSentence.split()
			englishSentence.append(self.null)
			foreignSentence = foreignSentence.split()

			for e in englishSentence:
				sTotalE = 0
				for f in foreignSentence:
					sTotalE += self.translate[self.englishToIndex[e]][self.spanishToIndex[f]] 
				for f in foreignSentence:
					weightedCount = self.translate[self.englishToIndex[e]][self.spanishToIndex[f]]/sTotalE
					newTranslate[self.englishToIndex[e]][self.spanishToIndex[f]] += weightedCount
					self.totalF[self.spanishToIndex[f]] += weightedCount
		self.translate = newTranslate
		#print self.translate

	def Mstep(self):
		"""
			Runs the Maximization step of the IBM Model 1 algorithm
			i.e. normalizes each row
		"""
		for i in range(len(self.englishVocabulary)):
			for j in range(len(self.spanishVocabulary)):
				self.translate[i][j] = self.translate[i][j] / self.totalF[j]

	def findSentenceTranslationProb(self, englishSentence, foreignSentence):
		""" 
			Because we can view the multisum (over all alignments) of products of t(f|e) 
			as the product (over all f) of the sum (over all e) of t(f|e), 
			we improve the time cost over the definition to O(|E||F|) from O(|F||E|**|F|)
		"""
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