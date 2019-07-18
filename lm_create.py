import jieba
import json
import string
import cn2an
import re
def read_file(filename):
    corpus=[]
    with open(filename,'r',encoding='UTF-8') as file_object:
        for line in file_object:
            corpus.append(json.loads(line))
            #print(corpus[0]['content'][0:])
    return corpus
 
def digits2ch_assit(matched):
    value=matched.group('value')
    return cn2an.an2cn(value,'low')

def digits2ch(string_in):
    return re.sub('(?P<value>\d+\.?\d*)', digits2ch_assit, string_in)

if(__name__=='__main__'):
    corpus=read_file('dataset\\new2016zh\\news2016zh_train.json')

    test=digits2ch(corpus)
    test=jieba.lcut(test,cut_all=False,HMM=True)
    punct=' ·，。？！'+string.punctuation #第一个是空格，这个地方真是要命
    #print(string.digits)
    for c in punct:
        while c in test:
            test.remove(c)
    for emm in test:
        print(emm)