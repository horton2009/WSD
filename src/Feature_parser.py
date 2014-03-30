#-*-coding:utf-8-*-

from io import StringIO
from lxml import etree


def Feature_extract(lexelt,dirout,ftype,Cwnd,wnd,sparator):
    # lexelt: the string contain a segment of a XML file, including all sample of one word 
    # dirout: the directory of the train data, where the extracted features of each word 
    #         would be a single file 
    #

    #------------------------------- get the key word -------------------------------------
    root=etree.XML(lexelt)
    word = root.xpath("./@item")[0].encode('utf-8')
    print word

    fout = open(dirout+word,"w")

    instances = root.xpath("./instance")

    for instance in instances:
        #----------------- put the pre- and post- context into list -----------------------
        contexts = instance.xpath("./context/text()")
        for i,string in enumerate(contexts):
            contexts[i]=contexts[i].replace("\n","").encode("utf-8")
        #----------------------------- get the idsense attribute --------------------------
        if("train"==ftype):
            sense = instance.xpath("./answer/@senseid")[0].encode("utf-8")
        elif("test"==ftype):
            sense = instance.xpath("./@id")[0].encode("utf-8")

        #---------------------------- get the tokens & postags-----------------------------
        tokens = instance.xpath("./postagging/word/token/text()")
        for i, token in enumerate(tokens):
            tokens[i] = tokens[i].replace("\n","").encode("utf-8")

        postags = instance.xpath("./postagging/word/@pos")
        for i, pos in enumerate(postags):
            postags[i] = postags[i].replace("\n","").encode("utf-8")
        #---------------------------- get the index of key word ---------------------------
        precontext = ""
        index = 0
        embed_phrase = ""
        phrase_pos = ""
        for i,token in enumerate(tokens):
            if(token!=word):
                precontext = precontext+token
                if(len(precontext)>len(contexts[0])):
                    index = i                                # the key word is in a phrase
                    embed_phrase = tokens[i]
                    phrase_pos = postags[i]
                    break
            elif(len(precontext) < len(contexts[0])):        # in the precontext there is a same word
                precontext = precontext+token
            elif(len(precontext)==len(contexts[0])):         # find the key word's index
                index = i
                break
        #----------------- List the embeded phrase into tokens & postags --------------------
        if(""!=embed_phrase):
            subtokens = instance.xpath("./postagging/word/subword/token/text()")
            subpostags = instance.xpath("./postagging/word/subword/@pos")
            del tokens[index]
            del postags[index]
            for i,subtoken in enumerate(subtokens):
                subtokens[i] = subtokens[i].replace("\n","").encode("utf-8")
                tokens.insert(index+i,subtokens[i])
                postags.insert(index+i,subpostags[i])

            index = index + subtokens.index(word)
        #----------------------------- get Features of key word -----------------------------
        #------------- 1. W-wnd ~ Wwnd, T-wnd~Twnd
        scale = range(-wnd,wnd+1)
        del scale[wnd]
        FeatureString = ""
        length = len(tokens)
        for i in scale:
            if(i<0 and (index+i)<0):
                token = "NULL_HEAD"
                tag = "NULL_HEAD"
            elif(i>0 and (index+i)>=length):    # last one is length-1
                token = "NULL_TAIL"
                tag = "NULL_TAIL"
            else:
                token = tokens[index+i]
                tag = postags[index+i]

            FeatureString = FeatureString + "W" + str(i) + "=" + token + sparator
            FeatureString = FeatureString + "T" + str(i) + "=" + tag + sparator
        #-------------2. PW, PT
        if(""!=embed_phrase):
            FeatureString = FeatureString + "PW" + "=" + embed_phrase + sparator
            FeatureString = FeatureString + "PT" + "=" + phrase_pos + sparator
        #-------------3. Content Word
        scale = range(-Cwnd,Cwnd)
        for i in scale:
            if(i<0 and (index+i)<0):
                pass
            elif(i>0 and (index+i)>=length):    # last one is length-1
                pass
            else:
                if(postags[index+i] in ["n","v"]):
                    FeatureString = FeatureString + tokens[index+i] + sparator
        #-------------4. Sense
        FeatureString = FeatureString + sense +"\n"

        fout.write(FeatureString)

    fout.close( )
    print "Finished :",word
    return word


def test():
    fin=open("./test.xml")
    dirout = "../train/test/"
    lexelt=fin.read()
    Feature_extract(lexelt,dirout)

if __name__=="__main__":
    test()