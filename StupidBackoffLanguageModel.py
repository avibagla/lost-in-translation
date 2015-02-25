import math, collections

def fDiv(a, b):
    return float(a)/float(b)

class CustomLanguageModel:
 
  def __init__(self):
    """Initialize your data structures in the constructor."""
    self.startToken = "<^>"
    self.UNK = "<NEIZVESTNOE>"
    self.d1 = 0.75
    self.d2 = 0.75
    self.discount = 0.75
    self.discountForOne = 0.5
    self.trigramCounts = collections.defaultdict(lambda: 0)
    self.bigramCounts  = collections.defaultdict(lambda: 0)
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.unigramsBefore = collections.defaultdict(lambda: set())
    self.unigramsAfter  = collections.defaultdict(lambda: set())
    self.total = 0
    self.backoffPenalizationFactor = math.log(0.4)
    # self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    total = 0
    for sentence in corpus:
        nextToLastToken = self.startToken
        lastToken = self.startToken
        sentence = sentence.split()
        for word in sentence:  
            token = word
            if nextToLastToken != self.startToken: 
                self.trigramCounts[(nextToLastToken, lastToken, token)] += 1
                #self.bigramsBefore[token].add((nextToLastToken, lastToken))
                self.unigramsAfter[(nextToLastToken, lastToken)].add(token)
            if lastToken != self.startToken: 
                self.bigramCounts[(lastToken, token)] += 1
                self.unigramsBefore[token].add(lastToken)
                self.unigramsAfter[lastToken].add(token)                
            self.unigramCounts[token] += 1
            self.total += 1
            nextToLastToken, lastToken = lastToken, token
    y = fDiv(len(self.unigramCounts), len(self.unigramCounts) + 2*len(self.bigramCounts))
    self.d1 = 1.0 - 2*fDiv(y*len(self.bigramCounts), len(self.unigramCounts))
    self.d2 = 2 - 3*fDiv(y*len(self.trigramCounts), len(self.bigramCounts))

  def unigramProbability(self, unigram):
    count = 1
    if unigram in self.unigramCounts:
        count = self.unigramCounts[unigram] + 1
    return fDiv(count, self.total + len(self.unigramCounts) + 1)

  def bigramProbability(self, bigram):
    discount = self.d1
    (given, w) = bigram
    bigramTerm = 0
    if bigram in self.bigramCounts:
        if self.bigramCounts[bigram] == 1:
            discount = discount*self.discountForOne
        bigramTerm      = fDiv(max(self.bigramCounts[bigram] - discount, 0), self.unigramCounts[given])
        knLambda        = fDiv(discount*len(self.unigramsAfter[given]), self.unigramCounts[given])
        pContinuation   = fDiv(len(self.unigramsBefore[w]), len(self.bigramCounts))
        if bigramTerm > 0: 
            return bigramTerm
        else:
            return bigramTerm + knLambda*pContinuation
    else: 
        return self.unigramProbability(w)

  def trigramProbability(self, trigram):
    discount = self.d2
    (pozaproshlii, proshlii, w) = trigram
    given = (pozaproshlii, proshlii)
    trigramTerm = 0
    knLambda = 1.0
    if trigram in self.trigramCounts:
        if self.trigramCounts[trigram] == 1:
            discount = discount*self.discountForOne
        trigramTerm     = fDiv(max(self.trigramCounts[trigram] - discount, 0), self.bigramCounts[given])
        if trigramTerm > 0: return trigramTerm
    if given in self.bigramCounts:
        knLambda        = fDiv(discount*len(self.unigramsAfter[given]), self.bigramCounts[given])
    pContinuation       = self.bigramProbability((proshlii, w))

    return trigramTerm + knLambda*pContinuation



  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    lastToken = self.startToken
    nextToLastToken = self.startToken
    for token in sentence: # iterate over words in the sentence
        if token not in self.unigramCounts:
            token = self.UNK
        if lastToken == self.startToken: 
            nextToLastToken, lastToken = lastToken, token
            continue
        thisscore = math.log(self.trigramProbability((nextToLastToken, lastToken, token)))
        #print thisscore
        score += thisscore

        nextToLastToken, lastToken = lastToken, token
    return score