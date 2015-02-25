import math, collections

class StupidBackoffLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.total = 0
    self.vocabulary = collections.defaultdict(lambda: 0)
    self.bigrams = collections.defaultdict(lambda: 0)
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    for sentence in corpus.corpus:
      for datum in sentence.data:  
        token = datum.word
        self.vocabulary[token] = self.vocabulary[token] + 1
        self.total += 1
      for i in xrange(1, len(sentence)-1):
        bigram = sentence.data[i-1].word + ' ' + sentence.data[i].word
        self.bigrams[bigram] = self.bigrams[bigram] + 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0 
    v = len(self.vocabulary)
    t = self.total
    for i in xrange(1, len(sentence)-1):
      bigram = sentence[i-1] + ' ' + sentence[i]
      count = self.bigrams[bigram]
      if count > 0:
        score += math.log(count)
        score -= math.log(self.vocabulary[sentence[i-1]])
      #take care of unknown bigrams
      if count == 0:
        c = self.vocabulary[sentence[i]]
        if c > 0:
          score += math.log(0.4)
          score += math.log(c + 1)
          score -= math.log(t + v)
          #take care of unknown words
        if c == 0:
          score += math.log(0.4)
          score += math.log(1)
          score -= math.log(t + v)
    return score
