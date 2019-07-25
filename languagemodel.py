import json

'''魔鬼语言模型~'''
class Language_Model():
    def __init__(self):
        self.path='lan_model\\'
        self.load_languagemodel()#产生self——unig,big,trig,lexicon
        pass

    def load_languagemodel(self):

        with open(self.path+'unigram.txt','r',encoding='UTF-8') as file_object:
            unig_list=file_object.readlines()#readlines好像不会有空行
            print(unig_list[-2])
            print(unig_list[-1])
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
                        self.lexicon[pinyin].append(temp[0])
                    else:
                        self.lexicon[pinyin]=[temp[0]]
                else :print('嘿嘿')

    def decode(self,list_pinyin):
        len_pinyin=len(list_pinyin)
        #构造词网格.....
        for from_index in range(0,len_pinyin):
            for to_index in range(from_index,len_pinyin):
                word_list=get_word(list_pinyin[from_index:to_index])
                if not word_list:continue
                
    def get_word(self,list_pinyin_word):
        pinyin=' '.join(list_pinyin_word)
        if self.lexicon.__contains__(pinyin):
            word_list=self.lexicon[pinyin]
            return word_list
        return None

if __name__=='__main__':
    LM=Language_Model()

    LM.decode(['lv4','shi4','yang2','chun1','yan1','jing3','da4','kuai4','wen2','zhang1','de5','di3',
               'se4','si4','yue4','de5','lin2','luan2','geng4','shi4','lv4','de5','xian1','huo2','xiu4','mei4','shi1',
              'yi4','ang4','ran2'])