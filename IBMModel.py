# encoding: utf-8

import time #for debugging
from collections import defaultdict, namedtuple #<3 dem defaults
from numpy import * #fast as lightning
import cPickle as pickle #file reading for python data in clean way
import nltk #nlp awesomesauce
import datetime
from StupidBackoffLanguageModel import CustomLanguageModel as lm
from Queue import PriorityQueue as pq
import re
import os
# from nltk.corpus import reuters
from nltk.tag.stanford import POSTagger
from pos_matching import ESTagToPOS, ENTagToPOS # Imports maps for EN and ES POS to a 'universal' code

etagger = POSTagger('../stanford-postagger/models/english-left3words-distsim.tagger', '../stanford-postagger/stanford-postagger.jar', encoding='utf8') 
stagger = POSTagger('../stanford-postagger/models/spanish-distsim.tagger', '../stanford-postagger/stanford-postagger.jar', encoding='utf8') 

englishCorpusFile = './es-en/train/europarl-v7.es-en.en' #'./es-en/train/small.en' #
spanishCorpusFile = './es-en/train/europarl-v7.es-en.es' #'./es-en/train/small.es' #

# Parts of Speech (POS) tagged
PosTaggedEnglishCorpusFile = './es-en/train/europarl-tagged.en.pickle'
PosTaggedSpanishCorpusFile = './es-en/train/europarl-tagged.es.pickle'
penaltyForTranslatingToDifferentPOS = .275
lmWeight = .00
tmWeight = 1.-lmWeight

# Will have some quirks, but overall okay
wordRegex = re.compile(r"^(?:[^\W\d_]|')+$", re.UNICODE)

#ex. format (spanish literally translated to english - fluent english translation)

# C -> CONJ						{CC , IN}
# Z -> NUM						{cd, }
# A -> ADJ						{JJ, JJR, JJS, LS, PDT}
# V -> verb or adverb (ex. exited floating - floated out)	{MD, VB, VBD, VBG, VBN, VBP, VBZ}
# N -> nouns or adj (ex. I have thirst - I am thirsty)		{NN, NNP, NNPS, NNS}
# P, DP -> pronoun				{PRP, PRP$}
# R -> ADV or verbs (ex. exited floating - floated out) 	{RB, RBR, RBS, }
# S -> preposition 				{RP}
# I -> interjection 			{UH}
# DT, DD -> wh 					{WDT, WP, WP$, WRB}
# DI,DA -> DET 					{DT}
# foreign word					{FW}
# F (punc)

# ENPOS = [
# 	CONJ 	= set([CC , IN]),
# 	NUM 	= set([CD]),
# 	ADJ 	= set([JJ, JJR, JJS, LS, PDT]),
# 	VERB 	= set([MD, VB, VBD, VBG, VBN, VBP, VBZ]),
# 	NOUN 	= set([NN, NNP, NNPS, NNS]),
# 	PRON 	= set([PRP, PRP$]),
# 	ADV 	= set([RB, RBR, RBS]),
# 	PREP 	= set([RP]),
# 	UH		= set([UH]),
# 	WH 		= set([WDT, WP, WP$, WRB]),
# 	DET 	= set([DT])
# ]

class IBM_Model_1:
	"""Class which trains IBM Model 1 and allows for testing"""

	def __init__(self):
		"""Sets initial variables for our algorithm"""
		self.PosTaggedEnglishCorpus = pickle.load(open(PosTaggedEnglishCorpusFile, "rb"))
		self.PosTaggedSpanishCorpus = pickle.load(open(PosTaggedSpanishCorpusFile, "rb"))

		self.englishVocabulary = set()
		self.spanishVocabulary = set()
		self.null = ("<<NULL>>", "NULL")
		self.lm = lm()
		for i in range(len(self.PosTaggedEnglishCorpus)):
			self.englishVocabulary |= set(self.parseTagsInSentenceNoPunc(self.PosTaggedEnglishCorpus[i], ENTagToPOS))
			self.spanishVocabulary |= set(self.parseTagsInSentenceNoPunc(self.PosTaggedSpanishCorpus[i], ESTagToPOS))
		self.englishVocabulary.add(self.null)
		print len(self.englishVocabulary)
		print len(self.spanishVocabulary)

	def parseTagsInSentenceNoPunc(self, taggedSentence, tagReducer):
		filteredSent = filter(lambda token: wordRegex.match(token[0]) is not None, taggedSentence)
		return [ ( tag[0].lower(), tagReducer(tag[1]) ) for tag in filteredSent ]

	def parseTagsInSentence(self, taggedSentence, tagReducer):
			return [ ( tag[0].lower(), tagReducer(tag[1]) ) for tag in taggedSentence ]

	def trainLM(self):
		self.lm.train() #self.englishCorpus
		pass

	def train(self, iterations):
		"""
			Trains our IBM Model 1 by iterating through the E step and the M step.
			EM will run at minimum one time, and at most 'iterations' times.
			Trains our language model on the English Corpus
		"""
		start = time.clock()
		self.englishToIndex = {}
		self.spanishToIndex = {}
		self.indexToEnglish = ['']*len(self.englishVocabulary)
		self.indexToSpanish = ['']*len(self.spanishVocabulary)
		countInv = 1./(len(self.spanishVocabulary))
		self.translate = empty( [len(self.englishVocabulary),len(self.spanishVocabulary)], dtype = float64 )
		self.translate.fill(countInv)

		#train the language model on English
		self.trainLM()

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
		zippedCorpuses = zip(self.PosTaggedEnglishCorpus, self.PosTaggedSpanishCorpus)
		
		for i in xrange(1, iterations+1):
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
			All sentence must already be tokenized
			i.e. 
			for each sentence pair
				for each e in E and f in F
					Add the total contribution of the sentences E,F to the word pair: tranlate(e,f)
		"""
		differentPOSPenalty = 0.8
		newTranslate = 	zeros((len(self.englishVocabulary),len(self.spanishVocabulary)), dtype = float64)
		self.totalF = 	zeros(len(self.spanishVocabulary), dtype = float64)

		for englishSentence, foreignSentence in trainingPhrases:
			englishSentence = self.parseTagsInSentenceNoPunc(englishSentence, ENTagToPOS) + [self.null]
			foreignSentence = self.parseTagsInSentenceNoPunc(foreignSentence, ESTagToPOS)

			for e in englishSentence:
				sTotalE = 0
				eidx = self.englishToIndex[e]
				for f in foreignSentence:
					multiplier = 1.
					if f[1]!= self.indexToEnglish[eidx][1] and self.indexToEnglish[eidx] != self.null:
						multiplier = differentPOSPenalty
					sTotalE += self.translate[eidx, self.spanishToIndex[f]] 
				for f in foreignSentence:
					sidx = self.spanishToIndex[f]
					weightedCount = self.translate[eidx, sidx]/sTotalE
					if f[1]!= self.indexToEnglish[eidx][1] and self.indexToEnglish[eidx] != self.null:
						weightedCount *= differentPOSPenalty
					newTranslate[eidx, sidx] += weightedCount
					self.totalF[sidx] += weightedCount
		self.translate = newTranslate

	def Mstep(self):
		"""
			Runs the Maximization step of the IBM Model 1 algorithm
			i.e. normalizes each col
		"""
		self.translate = self.translate / self.totalF[newaxis,:] #makes use of broadcasting ^_^

	def generateKBestFromTM(self, k, foreignSentence):
		# Best-first search of translations in the translation model
		# Returns the top K translations as judged by the translation model
		generatedSentences = []
		sentenceQueue = pq()
		logprob = self.getTMSentenceTransLogProbFromNth(foreignSentence, [0]*len(foreignSentence))
		sentenceQueue.put((-logprob, [0]*len(foreignSentence)))
		while (not sentenceQueue.empty()) and (len(generatedSentences) <= k):
			ithLogProb, ithWordIndices = sentenceQueue.get()
			if (ithWordIndices, -ithLogProb) not in generatedSentences:
				generatedSentences.append( (ithWordIndices, -ithLogProb) )


			for j in xrange(len(foreignSentence)): # And produce a new candidate for each sentence
				f = foreignSentence[j]
				if 	(f not in self.translationDictionary) or \
						(ithWordIndices[j] == len(self.translationDictionary[f]) - 1): 
						continue # No use checking these cases
				newCandidateIndices = list(ithWordIndices)
				newCandidateIndices[j] += 1

				logprob = self.getTMSentenceTransLogProbFromNth(foreignSentence, newCandidateIndices)
				sentenceQueue.put((-logprob, newCandidateIndices))

		for i in xrange(len(generatedSentences)):
		 	nthBest, logProb = generatedSentences[i]
			generatedSentences[i] = (self.englishSentenceUsing(foreignSentence, nthBest), logProb)

		return generatedSentences

	def englishSentenceUsing(self, foreignSentence, nthBest):
		englishSentence = []
		for j in xrange(len(foreignSentence)):
			e = f = foreignSentence[j]
			if f in self.translationDictionary:
				e = self.translationDictionary[f][nthBest[j]][0]
			if e != self.null:
				englishSentence.append(e)
		return englishSentence #" ".join(englishSentence)

	def taggedToSentence(self, taggedList):
		sentence = [taggedPair[0] for taggedPair in taggedList]
		return " ".join(sentence)

	def tokenize(self, sentence):
		return sentence.lower().split()

	def removeNonWords(self, sentence):
		return filter(lambda token: wordRegex.match(token) is not None, sentence)

	def tokenizeNoPunc(self, sentence):
		return self.removeNonWords(self.tokenize(sentence))

	def getTMSentenceTransLogProbFromEnglish(self, foreignSentence, englishSentence):
		# Sentences must be same length (including <<NULL>>'s)
		totalLogProb = 0
		for i in xrange(len(foreignSentence)):
			f = foreignSentence[i]
			if f not in self.translationDictionary: continue # only calculate words we've seen
			e = englishSentence[i]
			logProb = 1
			for index, (word, logProb) in enumerate(self.translationDictionary[f]):
				if word == e:
					break
			if logProb > 0: logProb = -inf
			totalLogProb += logProb
		return totalLogProb

	def getTMSentenceTransLogProbFromNth(self, foreignSentence, nthBest):
		# Sentences must be same length (including <<NULL>>'s)
		totalLogProb = 0
		for i in xrange(len(foreignSentence)):
			f = foreignSentence[i]
			if f in self.translationDictionary: # only calculate words we've seen
				originalPOS = foreignSentence[i][1]
				translation = self.translationDictionary[foreignSentence[i]][nthBest[i]]
				penaltyForTranslatingToDifferentPOS
				totalLogProb += translation[1]
				if originalPOS != translation[0][1] and translation[0] != self.null:
					totalLogProb -= penaltyForTranslatingToDifferentPOS
		return totalLogProb

	def predict(self, inputSentence):
		"""
			Takes in a foreign sentence and uses predictions to determine the 
			highest maximum liklihood sentence. Goes word by word and does a direct translation.
			Pre-condition: IBM Model training step is completed
			Pre-condition: Translation dictionary must be built before this method is called
		"""
		inputWords = self.parseTagsInSentence(inputSentence, ESTagToPOS)
		topk = self.generateKBestFromTM(10, inputWords)
		for i,(sentence, tmLogProb) in enumerate(topk):
			sentence = self.swapNounAdjConstructions(sentence)
		 	lmLogProb = self.getLMSentenceLogProb(self.taggedToSentence(sentence))
		 	topk[i] = (sentence, lmWeight*lmLogProb + tmWeight*tmLogProb, lmLogProb, tmLogProb)
		topk = sorted(topk, key=lambda sentenceAndLogProb: -sentenceAndLogProb[1])

		returnSent = self.taggedToSentence(topk[0][0])
		return returnSent

	def swapNounAdjConstructions(self, sentence):

		for i in xrange(len(sentence) - 1):
			word, tag =	sentence[i]
			if tag == "NOUN" and sentence[i+1][1] == "ADJ":
				sentence[i], sentence[i+1] = sentence[i+1], sentence[i]
		return filter(lambda token: token[0] != u'\xbf', sentence)


	def buildTranslationDictionary(self):
		print "Building translation dictionary"
		start = time.clock()
		self.translationDictionary = {}
		for i in xrange(5):
			bestEnglishTranslation = argmax(self.translate, axis=0)
			for spanishWord in self.spanishVocabulary:
				# Save the english word and translation probability
				sidx = self.spanishToIndex[spanishWord]
				eidx = bestEnglishTranslation[sidx]
				wordAndLogProb  = (self.indexToEnglish[int(eidx)], math.log(self.translate[eidx][sidx]) if self.translate[eidx][sidx] > 0 else -inf)
				self.translate[eidx][sidx] = 0. # Now set this to 0 so we can find new argmax
				if spanishWord not in self.translationDictionary:
					self.translationDictionary[spanishWord] = []
				self.translationDictionary[spanishWord].append(wordAndLogProb)
		print "Finished building translationDictionary", time.clock() - start

	def saveTranslationToFile(self, translationFileName='translation_'+time.strftime("%Y.%m.%d|%H.%M")):
		"""
			Since the EM algorithm only needs to be run once, this method uses python's builtin module
			Pickle to save the translation dictionary to a file.
			Pre-condition: Must be run after the translation system is trained
		"""
		#translationFileName = 'translation_'+time.strftime("%Y.%m.%d|%H.%M")
		if not hasattr(self, 'translationDictionary'):
			self.buildTranslationDictionary()
		start = time.clock()
		print "Saving to File"
		with open(translationFileName, 'wb') as f:
			pickle.dump(self.translationDictionary, f)
		# pickle.dump(self.translate, open("translationProbabilities", 'wb'))
		# lastDump = {
		# 	'englishToIndex': self.englishToIndex,
		# 	'spanishToIndex': self.spanishToIndex,
		# 	'indexToEnglish': self.indexToEnglish,
		# 	'indexToSpanish': self.indexToSpanish
		# }
		# pickle.dump(lastDump, open('mappingWordsTo2dArray', 'wb'))
		print "File Saved", translationFileName

		return translationFileName

	def readInTranslation(self, file_name):
		"""
			Reads in a pickle dump to the translation dictionary
		"""
		self.translationDictionary = pickle.load(open(file_name, "rb"))

	def getLMSentenceLogProb(self, inputSentence):
		"""
			Returns log probability of the sentence returned
		"""
		return self.lm.score(inputSentence)


def loadList(file_name):
    """Loads text files as lists of lines."""
    """Taken from pa5"""
    with open(file_name) as f:
        l = [line.strip().decode('utf8') for line in f]
    f.close()
    return l


		


# translation_nltk_tokenize
# BLEU-1 score: 47.795641
# BLEU-2 score: 9.851895

# translation_lm02_tm98
# BLEU-1 score: 49.064567
# BLEU-2 score: 10.339403

# translation_no_punc vanilla
# BLEU-1 score: 50.912483
# BLEU-2 score: 11.267491

# Translation alignment.7 predict.1
# BLEU-1 score: 51.070620
# BLEU-2 score: 11.411416

# translation_no_punc tm.9 lm.1
# BLEU-1 score: 50.701649
# BLEU-2 score: 11.129942

# translation_no_punc tm.95 lm.05
# BLEU-1 score: 50.676917
# BLEU-2 score: 11.164579

# translation_no_punc tm.6 lm.4
# BLEU-1 score: 50.219566
# BLEU-2 score: 10.877386

# translation_no_punc equal weights
# BLEU-1 score: 50.064520
# BLEU-2 score: 10.831915
