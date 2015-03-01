from IBMModel import IBM_Model_1
from bleu_score import run as bleu
import time
import sys #for argument input
import cPickle as pickle

"""
	tr is a convenient interface into the IBM model and auxilliary tools
	Flags: 
		-train 	number-of-iterations
			creates an instance of IBM_Model_1 and trains it on a corpus, then 
			outputs the translation dictionary.
		-dict 	location-of-translation-dictionary
			uses an existing translation dictionary to translate a file from 
			spanish to english and saves it in 'machine_translated'
		-eval 	anything
			runs the bleu_score script to evaluate the BLEU score of the 
			'machine_translated' file
"""

usage = """
tr: usage
	python tr.py -train number-of-iterations
	python tr.py -dict "location-of-translation-dictionary"
	python tr.py -eval anything
"""

def tic():
	return time.clock()
def toc(start = 0):
	return time.clock() - start

def main():
	
	options = getArguments(sys.argv)
	print "Evaluation flags:",  options
	IBM_Model = None

	if "-train" in options or "-dict" or "-postProcess" in options:
		
		IBM_Model = IBM_Model_1()

	if "-train" in options:
		tTrain = tic()
		
		IBM_Model.train( int(options["-train"]) )
		
		print "Trained", toc(tTrain)
		
		tTrans = tic()

		if "-dict" not in options:
			options["-dict"] = 'translation_'+time.strftime("%Y.%m.%d|%H.%M")
		translationFileName = IBM_Model.saveTranslationToFile(options["-dict"])

		print "Saved translation", toc(tTrans)
	if "-dict" in options:
		tTrans = tic()
		IBM_Model.trainLM()
		translationFileName = options["-dict"]
		translator = IBM_Model.readInTranslation(translationFileName)

		#spanishDevFile = loadList("./es-en/dev/newstest2012.es")
		with open("./es-en/dev/newstest2012-tagged.es.pickle", "rb") as f:
			spanishDevFile = pickle.load(f)
		translationOutput = open("machine_translated", 'w')
		for sentence in spanishDevFile:
			translationOutput.write("%s\n"%IBM_Model.predict(sentence).encode('utf8'))
		translationOutput.close()
		print "Translated", toc(tTrans)
		
	if "-postProcess" in options:
		"""Here we put in the file of the machine translated work to be post processed.. Right now my non existant
		   Sample in this directory is getting the glory. Note it must be load listed before getting processed."""
		print "Post Processing"
		tPost = tic()
		translationOutput = loadList("machine_translated")
		translationOutput = IBM_Model.postProcess(translationOutput)
		f = open('machine_translated', 'wb')
		for sentence in translationOutput:
			f.write("%s\n"%sentence)
		f.close()
		print "processed", toc(tPost)

	if "-eval" in options:
		bleu("./es-en/dev/newstest2012.en", "machine_translated")


def loadList(file_name):
    """Loads text files as lists of lines."""
    """Taken from pa5"""
    with open(file_name) as f:
        l = [line.strip().decode('utf8') for line in f]
    f.close()
    return l

def getArguments(arguments):
	if len(arguments) % 2 == 0:
		print "tr: each flag specified must have a value"
		print usage
		return {}
	args = {}
	arguments.pop(0)
	while len(arguments)>0:
		flag = arguments.pop(0)
		val = arguments.pop(0)
		args[flag] = val
	return args

main()
