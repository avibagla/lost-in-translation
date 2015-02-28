from IBMModel import IBM_Model_1
from bleu_score import run as bleu
import time
import sys #for argument input

def tic():
	return time.clock()
def toc(start = 0):
	return time.clock() - start

def main():
	
	options = getArguments(sys.argv)
	print "Evaluation flags:",  options
	IBM_Model = None
	if "-train" in options or "-dict" in options:
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

		spanishDevFile = loadList("./es-en/dev/newstest2012.es")
		translationOutput = open("machine_translated", 'w')
		for sentence in spanishDevFile:
			translationOutput.write("%s\n"%IBM_Model.predict(sentence).encode('utf8'))
		translationOutput.close()
		
		print "Translated", toc(tTrans)

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
		print "error: each flag specified must have a value"
		return {}
	args = {}
	arguments.pop(0)
	while len(arguments)>0:
		flag = arguments.pop(0)
		val = arguments.pop(0)
		args[flag] = val
	return args

main()
