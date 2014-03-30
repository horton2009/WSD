#-*-coding:utf-8-*-

import threading
import Feature_parser
import util


#----------------------------------------- Feature Extractor ----------------------------------------
class Extractor(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.lexelt = ""
        self.names = []
    
    def extract(self,infile, dirout,ftype,Cwnd=7,Wnd_l=1,Wnd_r=1,sparator=" | "):
        '''
        Params:
            infile:  the file path of original corpus for train / test;
            dirout:  the directory where the file containing extracted features should be placed;
            ftype:   set "train" if you are going to extracte train data from "infile",
                     set "test"  if you are going to extracte test data from "infile";
            Cwnd:    the width of the Content Word ('v','n') window, for both left and right;
                     i.e. if you set 5, then the content words in the scale of [-5,+5] will be extracted
            Wnd_l:   the left window for Word Feature and it's POS tag Feature;
            Wnd_r:   the right window for Word Feature and it's POS tag Feature;
            sparator:a marker used to sparator different features.

        '''
        f = open(infile)
        line = f.readline()
        line = f.readline()
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
                line = f.readline()            
            line = f.readline() 

            name = Feature_parser.Feature_extract(self.lexelt, dirout,ftype,Cwnd,Wnd_l,Wnd_r,sparator)
            if("test"==ftype):
                self.names.append(name) 
        if("test"==ftype):
            namefile = dirout + "namefile"
            util.writeNames(namefile, self.names)
        f.close()


    

def main():
    extractor = Extractor()
    extractor.extract("../corpus/train_corpus.xml", "../train/", "train", 7, 1, 1, " | ")
    extractor.extract("../corpus/test_corpus.xml", "../test/", "test", 7, 1, 1," | ")

if __name__ == '__main__':
    main()
