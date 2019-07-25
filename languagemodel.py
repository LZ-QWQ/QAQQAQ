import json

'''魔鬼语言模型~'''
class Language_Model():
    def __init__(self,path):
        '''
        文件路径+\\(windows下)，文件名写好了哦
        '''
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

        with open(self.path+'lz_lexicon.txt','r',encoding='UTF-8') as file_object:#这个真是魔鬼啊
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
                else :print('嘿嘿')
        pass

    def decode(self,list_pinyin):
        len_pinyin=len(list_pinyin)
        #构造词网格.....
        #下面这是魔鬼操作，，感觉好智障
        Words=[]
        for i in range(0,len_pinyin):
            Words.append([])
        for from_index in range(0,len_pinyin):
            for to_index in range(from_index,len_pinyin):
                word_list=self.get_word(list_pinyin[from_index:to_index+1])
                if not word_list:continue
                Words[to_index]+=word_list
        
        start='<s>'
        end='</s>'
        Pros=[]#跟Words对应
        for i in range(0,len_pinyin):
            Pros.append([])
        for p in range(0,len_pinyin):
            for word in Words[p]:
                word_length=len(word)
                if p-word_length+1<=0:
                    temp_pro=self.pro([start,word])
                    Pros[p].append(temp_pro)
                elif p-word_length+1>0:
                    p_2=p-word_length+1
                    for word_2 

                else:
                    raise ValueError('词的长度有问题哇QAQ')

            pass
#这个地方有点迷茫，，，，
    def get_n_gram(self,num):

    def get_word(self,list_pinyin_word):

        pinyin=' '.join(list_pinyin_word)
        if self.lexicon.__contains__(pinyin):
            word_list=self.lexicon[pinyin]
            return word_list
        return None

    def pro(word_list):#对数概率哦
        word_len=len(word_list)
        word_temp=' '.join(word_list)
        if word_len==2:
            if self.big.__contains__(word_temp):
                return self.big[word_temp][0]
            else :
                return back_pro(word_list[0])+pro(word_list[1])
        elif word_len==3:
            if self.trig.__contains__(word_temp):
                return self.trig[word_temp][0]
            else:
                return back_pro(word_list[0:1+1])+pro(word_list[1:])
        else:
            raise ValueError('这个列表词数不对哦~QAQ')

    def back_pro(word_list):
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

if __name__=='__main__':
    LM=Language_Model('lan_model\\')

    LM.decode(['lv4','shi4','yang2','chun1','yan1','jing3','da4','kuai4','wen2','zhang1','de5','di3',
               'se4','si4','yue4','de5','lin2','luan2','geng4','shi4','lv4','de5','xian1','huo2','xiu4','mei4','shi1',
              'yi4','ang4','ran2'])