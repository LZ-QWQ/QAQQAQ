import jieba
import json
import string
import cn2an
import re
'''反正我就是用结巴分词了。。。'''
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

def count_word(corpus_list,filename):
    path='lan_model\\'
    len_co=len(corpus_list)
    punct=' “”《》·，。？！'+string.punctuation#第一个是空格
    unig={}
    big={}
    trig={}
    all_1=0
    all_2=0
    all_3=0
    with open(path+filename+'_1.json','w+',encoding='UTF-8') as file_object:

        unig=json.load(file_object)
        for uni in corpus_list:#一元
            if uni not in punct:
                if unig.__contains__(uni):
                    unig[uni]+=1
                else:unig[uni]=1
                all_1+=1
            
            if unig.__contains__('total'):
                unig['total']+=all_1
            else :unig['total']=all_1
        json.dump(unig,file_object,indent=4,sort_keys=True,ensure_ascii=False)

    with open(path+filename+'_2.json','a+',encoding='UTF-8') as file_object:
        for i in range(0,len_co):#二元
            if i<1:continue
            else:
                if corpus_list[i] not in punct and corpus_list[i-1] not in punct:
                    bi=corpus_list[i-1]+corpus_list[i]
                    if big.__contains__(bi):
                        big[bi]+=1
                    else:big[bi]=1
            pass
    with open(path+filename+'_3.json','a+',encoding='UTF-8') as file_object:
        for i in range(0,len_co):#三元
            if i<2:continue
            else:
                if corpus_list[i] not in punct and\
                corpus_list[i-1] not in punct and corpus_list[i-2] not in punct:
                    tri=corpus_list[i-2]+corpus_list[i-1]+corpus_list[i]
                    if trig.__contains__(tri):
                        trig[tri]+=1
                    else:trig[tri]=1
            pass
    pass

def del_punc(corpus_line):
    punct=' ·，。？！'+string.punctuation#第一个是空格
    for c in punct:
        while c in corpus_line:
            corpus_line.remove(c)
    return corpus_line

if(__name__=='__main__'):
    corpus=read_file('dataset\\new2016zh\\news2016zh_valid.json')

    for line in corpus:
        res=line['content']
        temp=digits2ch(res)
        #temp=del_punc(temp)好像不能删
        temp_l=jieba.lcut(temp,cut_all=False,HMM=True)
        count_word(temp_l,'lz_lm_1')

