# -*- coding: utf-8 -*-

import os
from collections import defaultdict

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
    total_n = len(answers)
    reNum_n = len(results)

    re_dict = file_to_dict(results)
    an_dict = file_to_dict(answers)
    correct = defaultdict(int)
    total = defaultdict(int)
    for k in an_dict:
        total[k] += len(an_dict[k])
        for i in range(len(an_dict[k])):
            correct[k] += 1 if an_dict[k][i] == re_dict[k][i] else 0
    correct_n = sum(correct.values())
    micro_avg = compute_micro_avg(total, correct)
    macro_avg = compute_macro_avg(total, correct)
    print "Total: %d    Finished: %d    Correct: %d    MicroAVG: %f    MacroAVG: %f" % (
        total_n, reNum_n, correct_n, micro_avg, macro_avg)


def compute_micro_avg(total, correct):
    micro_avg = float(sum(correct.values())) / sum(total.values())
    return micro_avg


def compute_macro_avg(total, correct):
    accuracy_sum = 0.0
    for k in total:
        accuracy_sum += float(correct[k]) / total[k]
    macro_avg = accuracy_sum / len(total)
    return macro_avg


def file_to_dict(file_obj):
    d = defaultdict(list)
    for line in file_obj:
        line.strip()
        if line:
            word_type, word, label = line.split(' ')
            d[word_type].append(label)
    return d




def test():
    names = readDir("../test/")
    for name in names:
        print name

if __name__=="__main__":
    test()