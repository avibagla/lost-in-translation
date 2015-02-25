import math, collections
import re 
from nltk.corpus import reuters

def fDiv(a, b):
    return float(a)/float(b)

def t(prob, txt):
  try:
    math.log(prob)
  except:
    print txt

wordRegex = re.compile(r"^(?:[^\W\d_]|')+$", re.UNICODE)

def removeNonWords(sentence):
    return filter(lambda token: wordRegex.match(token) is not None, sentence)

def tokenize(sentence):
    return sentence.lower().split()

class CustomLanguageModel:

    def __init__(self):
        """Initialize your data structures in the constructor."""
        self.nullToken = "<^>"
        self.trigramCounts = collections.defaultdict(lambda: 0)
        self.bigramCounts  = collections.defaultdict(lambda: 0)
        self.unigramCounts = collections.defaultdict(lambda: 0)
        self.total = 0
        self.backoffPenalizationFactor = math.log(0.4)

    def train(self):
        """ Takes a corpus and trains your language model. 
            Compute any counts or other corpus statistics in this function.
        """  
        total = 0
        for fileid in reuters.fileids():
            for sentence in reuters.sents(fileid):
                for i in xrange(len(sentence)):
                    sentence[i] = sentence[i].lower()
                #sentence = removeNonWords(sentence)
                nextToLastToken = self.nullToken
                lastToken = self.nullToken
                for token in sentence:
                    if nextToLastToken != self.nullToken: 
                        self.trigramCounts[(nextToLastToken, lastToken, token)] += 1
                    if lastToken != self.nullToken: 
                        self.bigramCounts[(lastToken, token)] += 1
                    self.unigramCounts[token] += 1
                    self.total += 1
                    nextToLastToken, lastToken = lastToken, token

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the 
            sentence using your language model. Use whatever data you computed in train() here.
        """
        score = 0.0
        lastToken = self.nullToken
        nextToLastToken = self.nullToken
        #sentence = removeNonWords(tokenize(sentence))
        sentence = tokenize(sentence)
        for token in sentence: # iterate over words in the sentence
            if (nextToLastToken, lastToken, token) in self.trigramCounts:
                score += math.log(self.trigramCounts[(nextToLastToken, lastToken, token)])
                score -= math.log(self.bigramCounts[(lastToken, token)])
            elif (lastToken, token) in self.bigramCounts:  
                score += math.log(self.bigramCounts[(lastToken, token)])
                score -= math.log(self.unigramCounts[lastToken])
                score -= self.backoffPenalizationFactor
            else: # Use add-1 for unigrams
                if token in self.unigramCounts:
                    score += math.log(self.unigramCounts[token] + 1)
                score -= math.log(self.total + len(self.unigramCounts) + 1)
                score -= self.backoffPenalizationFactor*2
            nextToLastToken, lastToken = lastToken, token
        return score


