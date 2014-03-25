# -*- coding: utf-8 -*-

import os

def readDir(indir):
    names = []
    for i in os.listdir(indir):
        names.append(i)
    return names

def readNames(namefile):
    print "Reading namefile..."
    fin = open(namefile,"r")
    string = fin.read()
    names = string.strip(" ").split(" ")
    fin.close()
    return names

def writeNames(namefile,names):
    print "Writting namefile.."
    string = ""
    fout = open(namefile,"w")
    for name in names:
        string = string +" "+name
    fout.write(string)
    fout.close()



def evaluate(resultfile,answerfile):
    refin = open(resultfile)
    anfin = open(answerfile)

    results = refin.readlines()
    answers = anfin.readlines()

    total = len(answers)
    reNum = len(results)

    l = min(total,reNum)
    count = 0.0
    for i in range(l):
        if(results[i].strip().split(" ")[-1]==answers[i].strip().split(" ")[-1]):
            count = count + 1.0
    rate = count / total
    print "Total :",total,"   Finished:",reNum,"   Correct:",count,"     Correct rate:",rate







def test():
    names = readDir("../test/")
    for name in names:
        print name

if __name__=="__main__":
    test()