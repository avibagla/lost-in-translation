import os
import cPickle as pickle
from nltk.tag.stanford import POSTagger

def main():
	etagger = POSTagger('../stanford-postagger/models/english-left3words-distsim.tagger', '../stanford-postagger/stanford-postagger.jar', encoding='utf8') 
	stagger = POSTagger('../stanford-postagger/models/spanish-distsim.tagger', '../stanford-postagger/stanford-postagger.jar', encoding='utf8') 
	with open("./es-en/test/newstest2013-tagged.en.pickle", "wb") as f:
		pickle.dump(unPicklePeck("./es-en/test/partials", "newstest2013"), f)


def loadList(file_name):
    """Loads text files as lists of lines."""
    """Taken from pa5"""
    with open(file_name) as f:
        l = [line.strip().decode('utf8') for line in f]
    f.close()
    return l
    
def tagLinesAndPickle(sents, tagger, filename):
	tagged = []
	for sentence in sents:
		tagged.append(tagger.tag(sentence.split()))
	with open(file_name, 'wb') as f:
		pickle.dump(tagged,f)

def overwriteTo(filename, contents):
	try: 
		os.remove(filename)
	finally:
		text_file = open(filename, 'w')
		text_file.write("%s" % str(contents))
		text_file.close()

# Just hard code in the file names for now
def tagCorpusInChunks(sentences, tagger):
	stepSize = 100
	startAt = int(loadList("./es-en/test/partials/currentNTagged")[0])
	for i in xrange(startAt, len(spanishSents), stepSize):
		print "Starting chunk of 100 sentences from sentence", i
		sents = spanishSents[i: i+stepSize]
		filename = "./es-en/test/partials/newstest2013_es_" + str(i)
		tagLinesAndPickle(sents, stagger, filename)
		overwriteTo("./es-en/test/partials/currentNTagged", i+100)

def unPicklePeck(folder, filestub):
	pickledFiles = [filename for filename in os.listdir(folder) if filestub in filename]
	pickledFiles.sort(key=lambda x: int(x.split("_")[2]))
	sents = []
	for filename in pickledFiles:
		with open(folder + "/" + filename, "rb") as f:
			for sentence in pickle.load(f):
				sents.append(sentence)
	return sents

main()

