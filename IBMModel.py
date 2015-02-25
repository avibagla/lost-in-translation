import time #for debugging
from collections import defaultdict #<3 dem defaults
from numpy import * #fast as lightning
import cPickle as pickle #file reading for python data in clean way
import sys #for argument input
import nltk #nlp awesomesauce
import datetime


englishCorpusFile = './es-en/train/europarl-v7.es-en.en' #'./es-en/train/small.en' #
spanishCorpusFile = './es-en/train/europarl-v7.es-en.es' #'./es-en/train/small.es' #


class IBM_Model_1:
	"""Class which trains IBM Model 1 and allows for testing"""


	def __init__(self):
		"""Sets initial variables for our algorithm"""
		self.englishCorpus = loadList(englishCorpusFile)
		self.spanishCorpus = loadList(spanishCorpusFile)
		self.englishVocabulary = set()
		self.spanishVocabulary = set()
		self.null = "<<NULL>>"
		for i, sentence in enumerate(self.englishCorpus):
			self.englishVocabulary |= set(self.englishCorpus[i].lower().split())
			self.spanishVocabulary |= set(self.spanishCorpus[i].lower().split())
		self.englishVocabulary.add(self.null)

	def train(self, iterations):
		"""
			Trains our IBM Model 1 by iterating through the E step and the M step.
			EM will run at minimum one time, and at most 'iterations' times.
		"""
		start = time.clock()
		self.englishToIndex = {}
		self.spanishToIndex = {}
		self.indexToEnglish = ['']*len(self.englishVocabulary)
		self.indexToSpanish = ['']*len(self.spanishVocabulary)
		countInv = 1./(len(self.spanishVocabulary))
		self.translate = empty( [len(self.englishVocabulary),len(self.spanishVocabulary)], dtype = float64 )
		self.translate.fill(countInv)

		#Create Auxilliary Mapping dictionaries so our multidimensional array can be kept small
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

		#Zip corpuses and iterate through E step and M step
		zippedCorpuses = zip(self.englishCorpus, self.spanishCorpus)
		start = time.clock()
		self.Estep(zippedCorpuses)
		print "1 E Step completed", time.clock() - start
		start = time.clock()
		self.Mstep()
		print "1 M Step completed", time.clock() - start
		for i in xrange(2, iterations+1):
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

		# Why Zeros? Because, if two words are never aligned in sentences, we know that their 
		# probability will hit zero eventually, so we speed up the process.
		newTranslate = 	zeros((len(self.englishVocabulary),len(self.spanishVocabulary)), dtype = float64)
		self.totalF = 	zeros(len(self.spanishVocabulary), dtype = float64)

		for englishSentence, foreignSentence in trainingPhrases:
			# Tokenize sentences and add NULL to englishSentence
			englishSentence = englishSentence.lower().split() + [self.null]
			foreignSentence = foreignSentence.lower().split()

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
		"""
			Takes in a foreign sentence and uses predictions to determine the 
			highest maximum liklihood sentence. Goes word by word and does a direct translation.
			Pre-condition: IBM Model training step is completed
			Pre-condition: Translation dictionary must be built before this method is called
		"""
		inputWords = inputSentence.split()
		finalSentence = ''
		for word in inputWords:
			word = word.lower()
			if word in self.translationDictionary:
				finalSentence += (self.translationDictionary[word]+' ' if self.translationDictionary[word] != self.null else '')
			else:
				finalSentence += word+' '
		return finalSentence[:-1]


	def buildTranslationDictionary(self):
		print "Building translation dictionary"
		start = time.clock()
		bestEnglishTranslation = argmax(self.translate, axis=0)
		self.translationDictionary = {}
		for spanishWord in self.spanishVocabulary:
			sidx = self.spanishToIndex[spanishWord]
			self.translationDictionary[spanishWord] = self.indexToEnglish[int(bestEnglishTranslation[sidx])]
		print "Finished building translationDictionary", time.clock() - start
		# 	maxProb = -inf
		# 	currentTranslation = ''
		# 	for eidx in range(len(self.translate)):
		# 		# print eidx, sidx
		# 		if self.translate[eidx, sidx] > maxProb:
		# 			currentTranslation = self.indexToEnglish[eidx]
		# 			maxProb = self.translate[eidx, sidx]
		# 	self.translationDictionary[spanishWord] = currentTranslation
		# print "Finished building translationDictionary", time.clock() - start

	def saveTranslationToFile(self):
		"""
			Since the EM algorithm only needs to be run once, this method uses python's builtin module
			Pickle to save the translation dictionary to a file.
			Pre-condition: Must be run after the translation system is trained
		"""
		translationFileName = 'translation_'+time.strftime("%Y.%m.%d|%H.%M")
		if not hasattr(self, 'translationDictionary'):
			self.buildTranslationDictionary()
		start = time.clock()
		print "Saving to File"
		pickle.dump(self.translationDictionary, open(translationFileName, 'wb'))
		# pickle.dump(self.translate, open("translationProbabilities", 'wb'))
		lastDump = {
			'englishToIndex': self.englishToIndex,
			'spanishToIndex': self.spanishToIndex,
			'indexToEnglish': self.indexToEnglish,
			'indexToSpanish': self.indexToSpanish
		}
		pickle.dump(lastDump, open('mappingWordsTo2dArray', 'wb'))
		print "File Saved", translationFileName

		return translationFileName

	def readInTranslation(self, file_name):
		"""
			Reads in a pickle dump to the translation dictionary
		"""
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
	#IBM_Model.train(5) 
	print "Saved", time.clock() - start
	#translationFileName = IBM_Model.saveTranslationToFile()
	translationFileName = "translation_2015.02.24|22.13"
	

	translator = IBM_Model.readInTranslation(translationFileName)
	spanishDevFile = loadList("./es-en/dev/newstest2012.es")
	translationOutput = open("machine_translated", 'wb')
	for sentence in spanishDevFile:
		translationOutput.write("%s\n"%IBM_Model.predict(sentence))
	translationOutput.close()
	
	


	# start = time.clock()
	# pool = [square(x) for x in xrange(10000**2)]
	# print time.clock() - start


	
	# pool = multiprocessing.Pool(processes=cpus)
	# print pool.map(square, xrange(1000))
	# # IBM_Model = IBM_Model_1()
	# # IBM_Model.train() 
	# print time.clock() - start


main()
