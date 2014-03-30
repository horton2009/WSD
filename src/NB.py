#!/usr/bin/python
#-*-coding:utf-8-*-

from numpy import *
import util
import time
import Feature_Extractor

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

    def learn(self):                            # trainMatrix :must be numpy.array, to +
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
            self.Pc_xMap[c] =  array([0.05]*numWords)    
            #////////////////////////////////////////////////////////////////////
        ctotal = {}                             # record the total none zero feature number of samples for each class
        for i in range(numTrainDocs):
            c = self.classVec[i]
            self.Pc_xMap[c] = self.Pc_xMap[c] + trainMatrix[i]  
            ctotal[c] = ctotal.get(c,0) + sum(trainMatrix[i])

        for c in set(self.classVec):
            self.Pc_xMap[c] =  self.Pc_xMap[c]/ctotal[c]
        print "Finished Learning."


    def _learn(self,trainMatrix,trainClasses):   # overloaded learn() for tune
        print "\nLearning ..."
        numTrainDocs = len(trainMatrix)
        numWords = len(trainMatrix[0])
        for c in set(trainClasses):
            self.PcMap[c] = (trainClasses.count(c))/float(numTrainDocs)
            self.Pc_xMap[c] = ones(numWords)    # ones(): return a array with elements of float type
        
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


    # -------------------- Use the model and context features to predict the word sense ---------------------
    def predict(self,trainDir,testDir,dirOut):
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
            self.learn()

            self.infile = testDir + name
            self.loadFeature("test")

            for i,tesfFeatures in enumerate(self.testList):
                testVec = self.bagOfWords2Vec(tesfFeatures)
                result = self.classify(testVec)
                fout.write(name+" "+self.testName[i]+" "+result+"\n")

        fout.close()

        return outfile      # for evaluate


    # ------------ Randomly fold the trainSet to get train and test set,and test the error rate -------------
    def tune(self,n):
        self.loadFeature()                  
        self.createVocabList()

        length = len(self.docList)
        tuneSet_length = int(length/n)
        trainingSet = range(length);
        tuneSet=[]                                                       # create test set of index
        for i in range(tuneSet_length):
            randIndex = int(random.uniform(0,len(trainingSet)))
            tuneSet.append(trainingSet[randIndex])
            del(trainingSet[randIndex])  

        trainMat=[]; trainClasses = []
        for docIndex in trainingSet:                                     # train the classifier (get probs) trainNB0
            trainMat.append(self.bagOfWords2Vec(self.docList[docIndex]))
            trainClasses.append(self.classVec[docIndex])

        self._learn(array(trainMat),trainClasses)

        print "\nTesting... "
        error_count = 0.
        for docIndex in tuneSet:
            vec2Classify = self.bagOfWords2Vec(self.docList[docIndex])
            result = self.classify(vec2Classify)
            if(result!=self.classVec[docIndex]):
                error_count = error_count + 1
        print "Finished Testing."   

        error_rate = error_count/tuneSet_length
        return error_rate

    # ---------------------------- Repeatly test the model and get a even error rate ---------------------------
    def Random_Cross_Validation(self,times,fold):
        results = []
        for i in range(times):
            results.append(self.tune(fold))
        even_error_rate = sum(results)/len(results)
        print "\nFor",times,"times",fold,"Random Cross Validation, the even error rate is:",even_error_rate
        return even_error_rate

#---------------------------------------------------------------------------------------------------------------  




def main():

    # --------------------------------- Tune the features selection --------------------------------------------
    '''
    names = util.readNames("../test/namefile")
    fout = open("../result/Tune_Result "+str(time.ctime()),"a")
    for name in names:
        infile = "../train/"+name
        nb = NaiveBayes(infile)
        result = name + ":" + str(nb.Random_Cross_Validation(100,6)) + "\n"
        fout.write(result)
    '''

#    Feature_Extractor.extract("../corpus/train_corpus.xml", "../train/","train",0,3," | ")
#    Feature_Extractor.extract("../corpus/test_corpus.xml", "../test/","test",0,3," | ")

    for name in Feature_Extractor.Names:
        print name
    nb2 = NaiveBayes()
    resultfile = nb2.predict("../train/", "../test/", "../result/")

    util.evaluate(resultfile,"../result/test_answer")




if __name__ == "__main__":
    main()