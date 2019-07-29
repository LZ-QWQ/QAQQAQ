import json

'''
魔鬼语言模型~
三元条件概率  Katz退避  SRILM训练（thchs30）
词网格 Viterbi解码 
再叠加了个beam search的感觉。（这个感觉好像作用不大）
代码逻辑爆炸混乱
'''
class Language_Model():
    def __init__(self,path):
        '''
        文件路径+\\(windows下)，文件名写好了哦
        '''
        self.beam=3#维特比解码每个词时保存路径数
        self.start='<s>'
        self.end='</s>'
        self.path=path
        self.load_languagemodel()#产生self——unig,big,trig,lexicon
        pass

    def load_languagemodel(self):

        with open(self.path+'unigram.txt','r',encoding='UTF-8') as file_object:
            unig_list=file_object.readlines()#readlines好像不会有空行
            len_ulist=len(unig_list)
            self.unig={}
            for i in range(1,len_ulist):
                temp=unig_list[i].split(' ')
                self.unig[temp[0]]=[float(temp[1]),float(temp[2][:-1])]#这里有个换行符

        with open(self.path+'bigram.txt','r',encoding='UTF-8') as file_object:
            big_list=file_object.readlines()
            len_blist=len(unig_list)
            self.big={}
            for i in range(1,len_blist):
                temp=big_list[i].split(' ')
                self.big[' '.join(temp[0:2])]=[float(temp[2]),float(temp[3][:-1])]#这里有个换行符   列表 索引是左闭右开，，
        
        #我怎么给三元模型上了个回退概率？？？QAQ
        with open(self.path+'trigram.txt','r',encoding='UTF-8') as file_object:
            trig_list=file_object.readlines()
            len_tlist=len(trig_list)
            self.trig={}
            for i in range(1,len_tlist):
                temp=trig_list[i].split(' ')
                self.trig[' '.join(temp[0:3])]=[float(temp[3]),float(temp[4][:-1])]#这里有个换行符

        #pypinyin转换的声调有问题，，，（或者说很奇怪），，，correct是用thchs30原文件修正的
        with open(self.path+'lz_lexicon_correct.txt','r',encoding='UTF-8') as file_object:#这个真是魔鬼啊
            temp_all=file_object.read()
            temp_list=temp_all.split('\n')#最后一个是''
            self.lexicon={}
            for temp in temp_list:
                if temp!='':
                    temp=temp.split(' ')
                    pinyin=' '.join(temp[1:])
                    if self.lexicon.__contains__(pinyin):
                        if  temp[0] not in self.lexicon[pinyin]:
                            self.lexicon[pinyin].append(temp[0])
                    else:
                        self.lexicon[pinyin]=[temp[0]]
                #else :print('嘿嘿')
        pass

    def decode(self,list_pinyin):#产生了self——Words,Pros
        len_pinyin=len(list_pinyin)
        #构造词网格.....其实是 字
        #下面这是魔鬼操作，，感觉好智障
        self.Words=[]#哭
        for i in range(0,len_pinyin):
            self.Words.append([])
        for from_index in range(0,len_pinyin):
            for to_index in range(from_index,len_pinyin):
                word_list=self.get_word(list_pinyin[from_index:to_index+1])
                if not word_list:continue
                self.Words[to_index]+=word_list
        

        self.Pros=[]#跟Words对应
        for i in range(0,len_pinyin):
            self.Pros.append([])
        for p in range(0,len_pinyin):
            for word in self.Words[p]:
                word_length=len(word)
                if p-word_length==-1:
                    temp_pro=self.pro([self.start,word])
                    temp_word=Words_pro_path(temp_pro,[word])
                    self.Pros[p].append(temp_word)
                elif p-word_length>-1:
                    self.deal_pro_three(word,p)
                else:
                    raise ValueError('词的长度有问题哇QAQ')
        
        Pros_end=[]
        for p_ends in self.Pros[len_pinyin-1]:#加上结尾符号，，
            if type(p_ends)==list:
                for p_end in p_ends :#因为保存了多个（三个）
                    word_list_temp=[p_end.path[-1],p_end.path[-2],self.end]
                    temp_pro=p_end.pro+self.pro(word_list_temp)
                    Pros_end.append(Words_pro_path(temp_pro,p_end.path))
            else:
                 word_list_temp=[p_ends.path[-1],p_ends.path[-2],self.end]
                 temp_pro=p_ends.pro+self.pro(word_list_temp)
                 Pros_end.append(Words_pro_path(temp_pro,p_ends.path))

        Pros_end.sort(key=lambda P:P.pro,reverse=True)#这里也有变成三倍之多
        print(' '.join(Pros_end[0].path)+'\t'+str(Pros_end[0].pro))

    def deal_pro_three(self,word3,p3):
        '''
        专门用来计算解码时的三元条件概率和（完整）的计算
        感觉是核心哦~
        '''
        p2=p3-len(word3)
        temp_words=[]
        for word2 in self.Words[p2]:#，，-1是因为索引从0开始
            p1=p2-len(word2)
            if p1==-1:
                temp_pro=self.pro([self.start,word2,word3])+self.pro([self.start,word2])#三元模型哎这个是
                temp_words.append(Words_pro_path(temp_pro,[word2,word3]))
            elif p1>-1:
                for word1 in self.Words[p1]:
                    index=self.Words[p1].index(word1)#因为是按顺序插入的所以可以直接获取索引，Words和Prods是一一对应的                    
                    temp_pro_path=self.Pros[p1][index]
                    if type(temp_pro_path)==list:#因为某些情况并没有保存多个路径,并不是列表
                        for i in range(0,len(temp_pro_path)):#因为这里保存了多个路径及概率，数量可能没有self.beam这么多
                            word0=self.Pros[p1][index][i].path[-1]
                            temp_pro=self.pro([word1,word2,word3])+self.pro([word0,word1,word2])+self.Pros[p1][index][i].pro#合并前面概率
                            temp_words.append(Words_pro_path(temp_pro,self.Pros[p1][index][i].path+[word2,word3]))#合并前面路径
                    else:
                            word0=self.Pros[p1][index].path[-1]
                            temp_pro=self.pro([word1,word2,word3])+self.pro([word0,word1,word2])+self.Pros[p1][index].pro#合并前面概率
                            temp_words.append(Words_pro_path(temp_pro,self.Pros[p1][index].path+[word2,word3]))#合并前面路径
            else:
                raise ValueError('词的长度有问题哇QAQ')
        temp_words.sort(key=lambda w:w.pro,reverse=True)
        self.Pros[p3].append(temp_words[0:self.beam])#保存多个有点类似beam search

    def get_word(self,list_pinyin_word):
        '''
        #获取拼音组对应的词组，可能不存在
        '''
        pinyin=' '.join(list_pinyin_word)
        if self.lexicon.__contains__(pinyin):
            word_list=self.lexicon[pinyin]
            return word_list
        return None

    def pro(self,word_list):
        '''
        计算词组的条件概率，3、2、1元 对数概率哦
        '''
        if type(word_list)!=list:
            word_list=[word_list]
        word_len=len(word_list)
        word_temp=' '.join(word_list)
        if word_len==1:
            return self.unig[word_temp][0]#理论上不可能不存在
        elif word_len==2:
            if self.big.__contains__(word_temp):
                return self.big[word_temp][0]
            else :
                return self.back_pro(word_list[0])+self.pro(word_list[1])
        elif word_len==3:
            if self.trig.__contains__(word_temp):
                return self.trig[word_temp][0]
            else:
                return self.back_pro(word_list[0:1+1])+self.pro(word_list[1:])
        else:
            raise ValueError('这个列表词数不对哦~QAQ')

    def back_pro(self,word_list):
        '''
        计算回退概率，依然对数
        '''
        if type(word_list)!=list:
            word_list=[word_list]
        word_len=len(word_list)
        word_temp=' '.join(word_list)
        if word_len==1:
            if self.unig.__contains__(word_temp):
                return self.unig[word_temp][1]
            else :
                return 0            
        elif word_len==2:
            if self.big.__contains__(word_temp):
                return self.big[word_temp][1]
            else :
                return 0   
        else:
            raise ValueError('这个列表词数不对哦~QAQ')

class Words_pro_path():
    def __init__(self,pro,path):
        self.pro=pro
        self.path=path

if __name__=='__main__':
    LM=Language_Model('lan_model\\')

    LM.decode(['lv4','shi4','yang2','chun1','yan1','jing3','da4','kuai4','wen2','zhang1','de5','di3',
               'se4','si4','yue4','de5','lin2','luan2','geng4','shi4','lv4','de5','xian1','huo2','xiu4','mei4','shi1',
              'yi4','ang4','ran2'])
    LM.decode(['jin1','tian1','tian1','qi4','zhen1','hao3'])
    LM.decode(['xi1','an1','dian4','zi3','ke1','ji4','da4','xue2'])#pypinyin把很多 子 都翻成 zi5
    LM.decode(['kao3', 'yan2', 'yan1', 'yu3', 'ci2', 'hui4'])
    LM.decode(['xi3','huan1','ni3','o2'])
    LM.decode(['chao1','xi3','huan1','ni3','o4'])
    LM.decode(['xian4','dai4','han4','yu3','ci2','dian3','zhen1','de5','hen3','hao3','yong4'])
    LM.decode(['ai3','nai3','yi1','sheng1','shan1','shui3','lv4'])