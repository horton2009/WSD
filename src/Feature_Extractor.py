#-*-coding:utf-8-*-

import threading
import time
import Queue
import Feature_parser
import util

q = Queue.Queue(40)
Names = []
mutex = threading.Lock()

#---------- Read train data(xml) for each <lexelt> unit, and put it into the Queue ------------------
class Reader(threading.Thread):
    def __init__(self, infile,sleeptime):
        threading.Thread.__init__(self)
        self.infile = infile
        self.lexelt = ""
        self.sleeptime = sleeptime
    
    def run(self):
        line = self.infile.readline()
        global q
        while line:
            if "</corpus>" in line:
                break
            tag = 0
            self.lexelt=""
            while line:
                if("<lexelt" in line):
                    self.lexelt = self.lexelt+line
                    tag = 1                         # start reading train data for a word
                elif("</lexelt" in line):
                    self.lexelt = self.lexelt+line
                    tag = 0                         # finish reading train data for a word 
                    break
                elif(1 == tag):
                    self.lexelt = self.lexelt+line
                line = self.infile.readline()            
            line = self.infile.readline()   
            q.put(self.lexelt)
            print "There are still",q.qsize(),"task in the queue."

            time.sleep(self.sleeptime)

# ------------ Extract the feature the features and write into the specified directory --------------
class Extracter(threading.Thread):

    def __init__(self,dirout,sleeptime,ftype,Cwnd,Wnd,sparator):

        threading.Thread.__init__(self)
        self.dirout = dirout
        self.ftype = ftype
        self.sleeptime = sleeptime
        self.Cwnd = Cwnd
        self.Wnd = Wnd
        self.sparator = sparator
    
    def run(self):
        global q
        global Names

        while True:
            i = 0
            while(0==q.qsize()):
                i = i+1
                if(i>10):
                    break
                time.sleep(self.sleeptime)
            if(i>10):
                break

            lexelt = q.get()
            q.task_done()

            name = Feature_parser.Feature_extract(lexelt, self.dirout,self.ftype,self.Cwnd,self.Wnd,self.sparator)
            mutex.acquire()
            if("test"==self.ftype):
                Names.append(name)
            mutex.release()

        print "finished in ",self.name


def extract(infile,dirout,ftype="train",Cwnd=10,Wnd=2,sparator=" | "):
    global Names
    f=open(infile)
    if("train"==ftype):
        tnum = 5
        sleeptime = 0.3
    elif("test"==ftype):
        tnum = 1
        sleeptime = 0.1

    tR = Reader(f,sleeptime)
    tE = []
    for i in range(tnum):
        tE.append(Extracter(dirout,sleeptime,ftype,Cwnd,Wnd,sparator))

    tR.start()    
    for i in range(tnum):
        tE[i].start()
      
    tR.join()
    print "Reader finished"  

    for i in range(tnum):
        tE[i].join

    if("test"==ftype):
        util.writeNames(dirout+"namefile", Names)

    

if __name__ == '__main__':
#    extract("../corpus/train_corpus.xml","../train/","train",0,2," | ")
    extract("../corpus/test_corpus.xml","../test/","test",0,2," | ")