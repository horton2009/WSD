from nltk.classify import scikitlearn
from collections import defaultdict
from nltk.probability import FreqDist, DictionaryProbDist, ELEProbDist, sum_logs

a=defaultdict(set)
b=["a","b","c","d","d"]
f=FreqDist(b)

a["horton"].add(8)

a["horton"].add(6)

print a["horton"]
print f.N()