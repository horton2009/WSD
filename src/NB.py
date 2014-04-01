#!/usr/bin/python
#-*-coding:utf-8-*-

from numpy import *
import util
import time
import random
from Feature_Extractor import Extractor

class NaiveBayes(object):

    def __init__(self, infile=""):
        super(NaiveBayes, self).__init__()
        self.infile = infile
        self.PcMap = {}                     # record the pre_propertions
        self.Pc_xMap = {}                   # record the conditional propertion

        self.docList = []                   # record the List of original feature list
        self.classVec = []
        self.VocabList = []

        self.testList = []
        self.testName = []


    def loadFeature(self,ftype="train"):
        if("train"==ftype):
            print "\nLoading Train Data..."
            self.docList = []                           # make sure the list is clear befor loading
            self.classVec = []
            fin = open(self.infile)
            line = fin.readline()
            while line:
                tokens = line.strip("\n").split(" | ")
                self.classVec.append(tokens[-1])
                del(tokens[-1])
                self.docList.append(tokens)
                line = fin.readline()
            fin.close()
            print "Finished Loading."

        elif("test"==ftype):
            self.testList = []                          # make sure the list is clear befor loading
            self.testName = []
            print "\nLoading Test Feature..."
            fin = open(self.infile)
            line = fin.readline()
            while line:
                tokens = line.strip("\n").split(" | ")
                self.testName.append(tokens[-1])
                del(tokens[-1])
                self.testList.append(tokens)
                line = fin.readline()
            fin.close()
            print "Finished Loading."

                     
    def createVocabList(self):
        vocabSet = set([])
        for document in self.docList:
            vocabSet = vocabSet | set(document) #union of the two sets
        self.VocabList = list(vocabSet)

    def setOfWords2Vec(self, inputSet):
        returnVec = [0]*len(self.VocabList)
        for word in inputSet:
            if word in self.VocabList:
                returnVec[self.VocabList.index(word)] = 1
        return returnVec

    def bagOfWords2Vec(self, inputSet):
        returnVec = [0]*len(self.VocabList)
        for word in inputSet:
            if word in self.VocabList:
                returnVec[self.VocabList.index(word)] =returnVec[self.VocabList.index(word)] + 1
        return returnVec    

    def learn(self,smooth_rate):                            # trainMatrix :must be numpy.array, to +
        print "\nLearning ..."
        trainMat=[]
        for doc in self.docList:                # 
            trainMat.append(self.bagOfWords2Vec(doc))
        trainMatrix = array(trainMat)
        
        numTrainDocs = len(trainMatrix)
        numWords = len(trainMatrix[0])
        for c in set(self.classVec):
            self.PcMap[c] = (self.classVec.count(c))/float(numTrainDocs)
            #////////////////////////////////////////////////////////////////////
            self.Pc_xMap[c] =  array([smooth_rate]*numWords)    
            #////////////////////////////////////////////////////////////////////
        ctotal = {}                             # record the total none zero feature number of samples for each class
        for i in range(numTrainDocs):
            c = self.classVec[i]
            self.Pc_xMap[c] = self.Pc_xMap[c] + trainMatrix[i]  
            ctotal[c] = ctotal.get(c,0) + sum(trainMatrix[i])

        for c in set(self.classVec):
            self.Pc_xMap[c] =  self.Pc_xMap[c]/ctotal[c]

        print "Finished Learning."


    def _learn(self,trainMatrix,trainClasses,smooth_rate):   # overloaded learn() for tune
        print "\nLearning ..."
        numTrainDocs = len(trainMatrix)
        numWords = len(trainMatrix[0])
        for c in set(trainClasses):
            self.PcMap[c] = (trainClasses.count(c))/float(numTrainDocs)
            #////////////////////////////////////////////////////////////////////
            self.Pc_xMap[c] =  array([smooth_rate]*numWords)    
            #////////////////////////////////////////////////////////////////////
        
        ctotal = {}                             # record the total none zero feature number of samples for each class
        for i in range(numTrainDocs):
            c = trainClasses[i]
            self.Pc_xMap[c] = self.Pc_xMap[c] + trainMatrix[i]  
            ctotal[c] = ctotal.get(c,0) + sum(trainMatrix[i])

        for c in set(trainClasses):
            self.Pc_xMap[c] =  self.Pc_xMap[c]/ctotal[c]
        print "Finished Learning."


    def classify(self,vec2Classify):
        result = {}
        for c in set(self.classVec):
            for  v in (vec2Classify * self.Pc_xMap[c]):
                if(v>0):
                    result[c] = result.get(c,0.) + log(v) 
            result[c] = result.get(c,0.) + log(self.PcMap[c])
        import operator
        result = sorted(result.iteritems(), key=operator.itemgetter(1),reverse=True)
        return result[0][0]                           # sorted result is in type like [(c1, p1),(c2,p2)...]

    def _classify(self,vec2Classify,trainClasses):
        result = {}
        for c in set(trainClasses):
            for  v in (vec2Classify * self.Pc_xMap[c]):
                if(v>0):
                    result[c] = result.get(c,0.) + log(v) 
            result[c] = result.get(c,0.) + log(self.PcMap[c])
        import operator
        result = sorted(result.iteritems(), key=operator.itemgetter(1),reverse=True)
        return result[0][0] 


    # -------------------- Use the model and context features to predict the word sense ---------------------
    def predict(self,trainDir,testDir,dirOut,smooth_rate=0.001):
        names = util.readNames(testDir+"namefile")
        if(0==len(names)):
            names = util.readDir(testDir)

        outfile = dirOut + "Result_NB_" + str(time.ctime())
        fout = open(outfile,"w")

        for name in names:
            print "Predicting",name
            infile = trainDir + name
            self.infile = infile
            self.loadFeature()
            self.createVocabList()
            self.learn(smooth_rate)

            self.infile = testDir + name
            self.loadFeature("test")

            for i,tesfFeatures in enumerate(self.testList):
                testVec = self.bagOfWords2Vec(tesfFeatures)
                result = self.classify(testVec)
                fout.write(name+" "+self.testName[i]+" "+result+"\n")

        fout.close()

        return outfile      # for evaluate


    # ------------ Randomly fold the trainSet to get train and test set,and test the error rate -------------
    def tune(self,n,smooth_rate):
        self.loadFeature()                  
        self.createVocabList()

        length = len(self.docList)
        print "Total length = ",length
        scale = range(length);
        tuneSet = []
        trainSet = []       
        for c in set(self.classVec):
            cNum = self.classVec.count(c)
            selectNum = cNum/n
            if(0==selectNum):
                selectNum = 1  
            cSet = []                                              # create test set of index
            for i in scale:
                if(c==self.classVec[i]):
                    cSet.append(i)
            random.shuffle(cSet)
            tuneSet.extend(cSet[0:selectNum])

        print "TuneSet length = ",len(tuneSet)
        for i in scale:
            if i not in tuneSet:
                trainSet.append(i)

        trainMat=[]; trainClasses = []
        for docIndex in trainSet:                                     # train the classifier (get probs) trainNB0
            trainMat.append(self.bagOfWords2Vec(self.docList[docIndex]))
            trainClasses.append(self.classVec[docIndex])

        self._learn(array(trainMat),trainClasses,smooth_rate)

        print "\nTesting... "
        right_count = 0.0
        for docIndex in tuneSet:
            vec2Classify = self.bagOfWords2Vec(self.docList[docIndex])
            result = self._classify(vec2Classify,trainClasses)
            if(result==self.classVec[docIndex]):
                right_count = right_count + 1
        print "Finished Testing."   

        right_rate = right_count/len(tuneSet)
        return right_rate

    # ---------------------------- Repeatly test the model and get a even error rate ---------------------------
    def Random_Cross_Validation(self,times,fold,smooth_rate):
        results = []
        for i in range(times):
            results.append(self.tune(fold,smooth_rate))
        right_rate = sum(results)/len(results)
        print "\nFor",times,"times",fold,"Random Cross Validation, the even right_rate is:",right_rate
        return right_rate

#---------------------------------------------------------------------------------------------------------------  




def main():

    smooth_rate = 0.0001

    '''
    #------------------------------For Feature Extractor------------------------------
    extractor = Extractor()
    extractor.extract("../corpus/train_corpus.xml", "../train/", "train", 7, 3, 2, " | ")
    extractor.extract("../corpus/test_corpus.xml", "../test/", "test", 7, 3, 2," | ")
    '''
    names = util.readNames("../test/namefile")
    

    '''
    #------------------------------For Random Validation-------------------------------
    fout = open("../result/Tune_Result "+str(time.ctime())+".csv","a")
    results = []
    for name in names:
        infile = "../train/"+name

        nb = NaiveBayes(infile)
        print "---------",name,"----------"
        result = nb.Random_Cross_Validation(20,4,smooth_rate)
        results.append(result)
        reStr = name + "," + str(result) + "\n"
        fout.write(reStr)
    fout.close()
    print "Macro AVG:",sum(results)/len(results)

    '''
    #------------------------------------For Test--------------------------------------
    for name in names:
        print name
    nb2 = NaiveBayes()
    resultfile = nb2.predict("../train/", "../test/", "../result/",smooth_rate)
    util.evaluate(resultfile, "../result/test_answer")
    
    

if __name__ == "__main__":
    main()