# Parts of speech matching file 

eCONJ 	= ("CC" , "IN")
eNUM 		= ("CD",)
eADJ 		= ("JJ", "JJR", "JJS", "LS", "PDT")
eVERB 	= ("MD", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "EX")
eNOUN 	= ("NN", "NNP", "NNPS", "NNS")
ePRON 	= ("PRP", "PRP$")
eADV 		= ("RB", "RBR", "RBS")
ePREP 	= ("RP","TO")
eUH			= ("UH",)
eWH 		= ("WDT", "WP", "WP$", "WRB")
eDET 		= ("DT",)

sCONJ = ("C",)
sNUM 	= ("Z",)
sADJ 	= ("A",)
sVERB = ("V",)
sNOUN = ("N",)
sPRON = ("P", "DP")
sADV 	= ("R",)
sPREP = ("S",)
sUH		= ("I",)
sWH 	= ("DT", "DD")
sDET 	= ("DI", "DA")

# As a dictionary
ENPOS = {
	eCONJ: 	"CONJ",
	eNUM:		"NUM",
	eADJ:		"ADJ",
	eVERB: 	"VERB",
	eNOUN: 	"NOUN",
	ePRON: 	"PRON",
	eADV: 	"ADV",
	ePREP: 	"PREP",
	eUH:		"UH",
	eWH: 		"WH",
	eDET: 	"DET",
}


ESPOS = {
	sCONJ: 	"CONJ",
	sNUM:		"NUM",
	sADJ:		"ADJ",
	sVERB: 	"VERB",
	sNOUN: 	"NOUN",
	sPRON: 	"PRON",
	sADV: 	"ADV",
	sPREP: 	"PREP",
	sUH:		"UH",
	sWH: 		"WH",
	sDET: 	"DET",
}

def expandDict(mymap):
	expanded = {}
	for part_of_speech, universal_code in mymap.items():
		for en_code in part_of_speech:
			expanded[en_code] = universal_code
	return expanded

def reduceESTag(tag):
	tag = tag.upper()
	if tag[0] != "D": return tag[0]
	else: return tag[:2]

ENPOS = expandDict(ENPOS)
ESPOS = expandDict(ESPOS)

def ESTagToPOS(tag):
	tag = reduceESTag(tag.upper())
	if tag in ESPOS: 	return ESPOS[tag]
	else: return "OTHER"

def ENTagToPOS(tag):
	if tag in ENPOS: 	return ENPOS[tag.upper()]
	else:							return "OTHER"
