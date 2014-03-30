#!/usr/bin/python
#-*-coding:utf-8-*-

import sys 
from locale import atof
import ANN
import ANN_loader
import util
import time

class bpTest(object):
    """docstring for bpTest"""
    def __init__(self, trainfile="",testfile=""):
        super(bpTest, self).__init__()
        self.trainfile = trainfile
        self.testfile = testfile

        self.VocabOrderList = []               # order for input list
        self.ClassOrderList = []               # order for output list

        self.docList = []                      # record the List of original feature list of train data
        self.ClassList = []

        self.testList = []                     # record the List of original feature list of test data
        self.testName = []

    def createVocabOrder(self):
        vocabSet = set([])
        for document in self.docList:
            vocabSet = vocabSet | set(document) # union of the two sets
        self.VocabOrderList = list(vocabSet)

    def createClassOderList(self):
        classSet = set([])
        for c in self.classList:
            classSet.add(c)
        self.ClassOrderList = list(classSet)

    def loadTrainData(self):
        print "\nLoading Train Data..."
        self.docList = []                           # make sure the list is clear befor loading
        self.classList = []
        fin = open(self.trainfile)
        line = fin.readline()
        while line:
            tokens = line.strip("\n").split(" | ")
            self.classList.append(tokens[-1])
            del(tokens[-1])
            self.docList.append(tokens)
            line = fin.readline()
        fin.close()
        print "Finished Loading."

    def loadTestData(self):
        self.testList = []                          # make sure the list is clear befor loading
        self.testName = []
        print "\nLoading Test Feature..."
        fin = open(self.testfile)
        line = fin.readline()
        while line:
            tokens = line.strip("\n").split(" | ")
            self.testName.append(tokens[-1])
            del(tokens[-1])
            self.testList.append(tokens)
            line = fin.readline()
        fin.close()
        print "Finished Loading."

    def setOfWords2Vec(self, inputSet):
        returnVec = [0.0]*len(self.VocabOrderList)
        for word in inputSet:
            if word in self.VocabOrderList:
                returnVec[self.VocabOrderList.index(word)] = 1.0
        return returnVec

    def bagOfWords2Vec(self, inputSet):
        returnVec = [0.0]*len(self.VocabOrderList)
        for word in inputSet:
            if word in self.VocabOrderList:
                returnVec[self.VocabOrderList.index(word)] =returnVec[self.VocabOrderList.index(word)] + 1.0
        return returnVec

    def class2Vec(self, outputClass):
        returnVec = [0.0]*len(self.ClassOrderList)
        returnVec[self.ClassOrderList.index(outputClass)] = 1.0
        return returnVec 

    def getExamplars(self):              # get train Data typt [([inList], [outList]), ([inLIst] ,[outList]) ....]
        self.loadTrainData()
        self.createVocabOrder()
        self.createClassOderList()
        examplars = []
        for i, doc in enumerate(self.docList):
            inputVec = self.bagOfWords2Vec(doc)
            outputVec = self.class2Vec(self.classList[i])      
            examplar = (inputVec,outputVec)
            examplars.append(examplar)                  
        return examplars

    def getInputs(self):                       # get test Data
        self.loadTestData()
        Inputs = []
        for i, doc in enumerate(self.testList):
            inputVec = self.bagOfWords2Vec(doc)
            Inputs.append(inputVec)
        return Inputs

    def predict(self,trainDir,testDir,dirOut):
        names = util.readNames(testDir+"namefile")
        if(0==len(names)):
            names = util.readDir(testDir)

        outfile = dirOut + "Result_ANN_" + str(time.ctime())
        fout = open(outfile,"w")

        for name in names:
            self.trainfile = trainDir + name
            self.testfile = testDir + name

            examplars = self.getExamplars()
            inputs = self.getInputs()

            outNum = len(self.ClassOrderList)
            inNum = len(self.VocabOrderList)
            hiddenNum =int(inNum/20)

            bpNet=ANN.BackPropNet()
            bpNet.addinput(inNum)
            bpNet.addhidden(5)
            bpNet.addouput(outNum)

            print "Learning",name
            bpNet.learn(examplars,40)
            print "Predict",name
            results = bpNet.run(inputs)
            print "Writting result..."
            for i, e in enumerate(self.testName):

                string = name + " " + e
                index = results[i].index(max(results[i]))
                string = string + " " + self.ClassOrderList[index] + "\n"
                fout.write(string) 
                fout.flush()
        fout.close()
        print "Finished Predicting"  
        return outfile


                    
                



def main():
    #trainfile = "../train/中医"
    #testfile = "../test/中医"
    #bpt = bpTest(trainfile, testfile)


    bpt = bpTest()
    outfile = bpt.predict("../train/","../test/","../result/")
    util.evaluate(outfile, "../result/test_answer")
    

if __name__=="__main__":
    main()
