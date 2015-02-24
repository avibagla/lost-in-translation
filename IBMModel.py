import time
from collections import defaultdict
from numpy import *
import cPickle as pickle

englishCorpusFile = './es-en/train/europarl-v7.es-en.en' #'../es-en/train/small.en'
spanishCorpusFile = './es-en/train/europarl-v7.es-en.es' #'../es-en/train/small.es'


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


	def train(self, iterations):
		start = time.clock()
		self.englishToIndex = {}
		self.spanishToIndex = {}
		self.indexToEnglish = ['']*len(self.englishVocabulary)
		self.indexToSpanish = ['']*len(self.spanishVocabulary)
		countInv = 1./(len(self.spanishVocabulary))
		self.translate = empty( [len(self.englishVocabulary),len(self.spanishVocabulary)], dtype = float64 )
		self.translate.fill(countInv)
		i = 0
		for word in self.englishVocabulary:
			self.englishToIndex[word] = i
			self.indexToEnglish[i] = word
			i+=1
		i = 0
		for word in self.spanishVocabulary:
			self.spanishToIndex[word] = i
			self.indexToSpanish[i] = word
			i+=1
		print "table is built", time.clock() - start

		zippedCorpuses = zip(self.englishCorpus, self.spanishCorpus)
		start = time.clock()
		self.Estep(zippedCorpuses)
		print "E Step completed", time.clock() - start
		start = time.clock()
		self.Mstep()
		print "M Step completed", time.clock() - start
		for i in xrange(1, iterations):
			start = time.clock()
			self.Estep(zippedCorpuses)
			print i, "E Step completed", time.clock() - start
			start = time.clock()
			self.Mstep()
			print i, "M Step completed", time.clock() - start
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
		newTranslate = 	zeros((len(self.englishVocabulary),len(self.spanishVocabulary)), dtype = float64)
		self.totalF = 	zeros(len(self.spanishVocabulary), dtype = float64)

		for englishSentence, foreignSentence in trainingPhrases:
			# Tokenize sentences and add NULL to englishSentence
			englishSentence = englishSentence.split() + [self.null]
			#englishSentence.append(self.null)
			foreignSentence = foreignSentence.split()

			for e in englishSentence:
				sTotalE = 0
				eidx = self.englishToIndex[e]
				for f in foreignSentence:
					sTotalE += self.translate[eidx, self.spanishToIndex[f]] 
				for f in foreignSentence:
					sidx = self.spanishToIndex[f]
					weightedCount = self.translate[eidx, sidx]/sTotalE
					newTranslate[eidx, sidx] += weightedCount
					self.totalF[sidx] += weightedCount
		self.translate = newTranslate

	def Mstep(self):
		"""
			Runs the Maximization step of the IBM Model 1 algorithm
			i.e. normalizes each col
		"""
		self.translate = self.translate / self.totalF[newaxis,:] #makes use of broadcasting ^_^

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
		inputWords = inputSentence.split()
		finalSentence = ''
		for word in inputWords:
			if word in self.translationDictionary:
				finalSentence += (self.translationDictionary[word]+' ' if self.translationDictionary[word] != self.null else '')
		return finalSentence[:-1]


	def buildTranslationDictionary(self):
		print "Building translation dictionary"
		start = time.clock()
		self.translationDictionary = {}
		for spanishWord in self.spanishVocabulary:
			sidx = self.spanishToIndex[spanishWord]
			maxProb = -inf
			currentTranslation = ''
			for eidx in range(len(self.translate)):
				# print eidx, sidx
				if self.translate[eidx, sidx] > maxProb:
					currentTranslation = self.indexToEnglish[eidx]
					maxProb = self.translate[eidx, sidx]
			self.translationDictionary[spanishWord] = currentTranslation
		print "Finished building translationDictionary", time.clock() - start

	def saveTranslationToFile(self):
		"""Must be run after the translation system is trained"""
		if translationDictionary not in self:
			self.buildTranslationDictionary()
		start = time.clock()
		print "Saving to File"
		pickle.dump(self.translationDictionary, open('translation_' + str(time.clock()), 'wb'))
		print "File Saved", time.clock() - start

	def readInTranslation(self, file_name):
		self.translationDictionary = pickle.load(open(file_name, "rb"))





def loadList(file_name):
    """Loads text files as lists of lines."""
    """Taken from pa5"""
    with open(file_name) as f:
        l = [line.strip() for line in f]
    f.close()
    return l



def main():
	start = time.clock()


	# pool = multiprocessing.Pool(processes=cpus)
	# pool.map(square, xrange(10000**2))
	IBM_Model = IBM_Model_1()
	# IBM_Model.train(100) 
	IBM_Model.readInTranslation("translation_4141.32095")

	spanishDevFile = loadList("./es-en/dev/newstest2012.es")
	translationOutput = open("machine_translated", 'wb')
	for sentence in spanishDevFile:
		translationOutput.write("%s\n"%IBM_Model.predict(sentence))
	translationOutput.close()
	print "Saved", time.clock() - start
	# IBM_Model.saveTranslationToFile()
	


	# start = time.clock()
	# pool = [square(x) for x in xrange(10000**2)]
	# print time.clock() - start


	
	# pool = multiprocessing.Pool(processes=cpus)
	# print pool.map(square, xrange(1000))
	# # IBM_Model = IBM_Model_1()
	# # IBM_Model.train() 
	# print time.clock() - start


main()
