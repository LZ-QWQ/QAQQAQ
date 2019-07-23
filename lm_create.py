import jieba
import json
import string
import cn2an
import re
'''反正我就是用结巴分词了。。。'''
'''反正这个是废掉了哦，自己做统计失败。。。'''
def read_file(filename):
    corpus=[]
    with open(filename,'r',encoding='UTF-8') as file_object:
        i=0
        for line in file_object:
            if i>300:break
            i+=1
            temp_dict=json.loads(line)
            corpus.append(temp_dict['content'])
            #print(corpus[0]['content'][0:])
    return corpus
 
def digits2ch_assit(matched):
    value=matched.group('value')
    if len(str(value))>16:return '一百'#有毒把？
    return cn2an.an2cn(value,'low')

def digits2ch(string_in):
    return re.sub('(?P<value>[0-9]\d+\.?[0-9]\d*)', digits2ch_assit, string_in)

def count_word_uni(corpus_list,unig={}):
    len_co=len(corpus_list)
    punct=' ９３“”…：—《》·，。？！'+string.punctuation#第一个是空格９３这是魔鬼
    case=string.ascii_letters
    all_1=0
    for uni in corpus_list:#一元
        if uni not in punct and uni[0] not in case:
             if unig.__contains__(uni):
                unig[uni]+=1
             else:unig[uni]=1
             all_1+=1
            
        if unig.__contains__('total'):
           unig['total']+=all_1
        else :unig['total']=all_1

def count_word_bi(corpus_list,filename):
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
    #corpus=read_file('dataset\\new2016zh\\news2016zh_train.json')

    #path='lan_model\\'
    #filename='lz_lm_1'
    #unig={}
    #big={}
    #trig={}


    #for line in corpus:
    #    print(line)
    #    temp=digits2ch(line)
        
    #    #temp=del_punc(temp)好像不能删
    #    temp_l=jieba.lcut(temp,cut_all=False,HMM=True)
    #    count_word_uni(temp_l,unig)
    #with open(path+filename+'_1.json','w+',encoding='UTF-8') as file_object:
    #    json.dump(unig,file_object,indent=4,sort_keys=True,ensure_ascii=False)


    #这个是处理thchs30里的lm文件,,,,就拿来用下啦,,,自己做太麻烦了吧！！！！
    path='lan_model\\'
    filename='word.3gram.lm'
#有点问题还要修复
    with open(path+filename,'r',encoding='UTF-8') as file_object:
        lines=file_object.readlines()
        len_wordlm=len(lines)
        all_1=int(lines[2][8:-1])
        all_2=int(lines[3][8:-1])
        all_3=int(lines[4][8:-1])
        with open(path+'unigram.txt','w',encoding='UTF-8') as file_object:
            file_object.write('unigram '+str(all_1)+'\n')
            for i in range(7,7+all_1):
                temp=lines[i].split('\t')
                if len(temp)==2:
                    out_str=temp[1][0:-1]+' '+temp[0]+' 0\n'
                else: out_str=temp[1]+' '+temp[0]+' '+temp[2][0:-1]+'\n'
                file_object.write(out_str)

        with open(path+'bigram.txt','w',encoding='UTF-8') as file_object:
            file_object.write('bigram '+str(all_2)+'\n')
            for i in range(7+all_1+2,7+all_1+2+all_2):
                temp=lines[i].split('\t')
                if len(temp)==2:
                    out_str=temp[1][0:-1]+' '+temp[0]+' 0\n'
                else: out_str=temp[1]+' '+temp[0]+' '+temp[2][0:-1]+'\n'
                file_object.write(out_str)

        with open(path+'trigram.txt','w',encoding='UTF-8') as file_object:    
            file_object.write('trigram '+str(all_3)+'\n')
            for i in range(7+all_1+2+all_2+2,7+all_1+2+all_2+2+all_3):
                temp=lines[i].split('\t')
                if len(temp)==2:
                    out_str=temp[1][0:-1]+' '+temp[0]+' 0\n'
                else: out_str=temp[1]+' '+temp[0]+' '+temp[2][0:-1]+'\n'
                file_object.write(out_str)
