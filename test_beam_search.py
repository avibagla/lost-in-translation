from heapq import *

NULL = "<<NULL>>"
englishTranslation = [("mary", -2), ("did", -1), ("not", -2), (NULL,-3),("slap",-3), ("the",-1), ("witch",-1), ("green",-0.5)]

def beamSearchStackDecoder(self, englishTranslation):
	stackSize = 100
	spanishPositions = set([i for i in xrange(len(englishTranslation))])
	hypothesisStack = [[] for i in xrange(len(englishTranslation) + 1)]
	hypothesisStack[0].append((0, [])) 
	for i in xrange(len(englishTranslation)):
		for prob, hyp in hypothesisStack[i]:
			try:
				remaining = spanishPositions - set(hyp)
			except: 
				remaining = spanishPositions
			for pos in remaining:
				newHyp = hyp + [pos]
				hypothesisCost 	= self.calculateSentenceLogProbability(newHyp, englishTranslation)
				futureCost = 0
				if len(hypothesisStack[i+1]) > stackSize: 
					heapreplace( hypothesisStack[i+1], (hypothesisCost + futureCost, newHyp) ) 
				else: 
					heappush( hypothesisStack[i+1], (hypothesisCost + futureCost, newHyp) )
	return max(hypothesisStack[-1])

def calculateSentenceLogProbability(positions, englishTranslation):
	# positions: 						list of foreign word positions
	# englishTranslation:		list of (englishWord, logProbOfTranslation)
	sentence = ""
	logProb = 0.0
	logDistortionPenalty =0.1

	# Translation model probability
	for position in positions:
		if englishTranslation[position][0] != NULL:
			sentence = sentence + " " + englishTranslation[position][0]
		logProb += englishTranslation[position][1]

	# Language model probability
	logProb += self.getLMSentenceLogProb(sentence)

	# Distortion model probability
	paddedPositions = [-1] + positions
	for i in xrange(1, len(paddedPositions)):
		logProb -= abs(paddedPositions[i] - paddedPositions[i-1]-1)*logDistortionPenalty
	
	return logProb

def getLMSentenceLogProb(sentence):
	sentence = sentence.split()
	logProb = 0
	for i in xrange(len(sentence) - 1):
		bigram = sentence[i] + " " + sentence[i+1]
		if bigram == "mary did": logProb += 0
		elif bigram == "did not":	logProb += 0
		elif bigram == "not slap": logProb += 0
		elif bigram == "slap the": logProb += 0
		elif bigram == "the green": logProb += 0
		elif bigram == "green witch": logProb += 0
		else: logProb += -10
	return logProb


prob, positions =  beamSearchStackDecoder(englishTranslation)
sentence = ""
for position in positions:
		if englishTranslation[position][0] != NULL:
			sentence = sentence + " " + englishTranslation[position][0]
print sentence
