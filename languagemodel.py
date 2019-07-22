import json

'''魔鬼语言模型~'''
class Language_Model():
    def __init__(self):
        self.path='lan_model\\'
        self.load_languagemodel()#产生self——unigram,bigram,trigram,dic

        pass

    def load_languagemodel(self):

        with open(self.path+'unigram.json','r',encoding='UTF-8') as file_object:
            self.unig=json.load(file_object)
        with open(self.path+'bigram.json','r',encoding='UTF-8') as file_object:
            self.big=json.load(file_object)
        with open(self.path+'trigram.json','r',encoding='UTF-8') as file_object:
            self.trig=json.load(file_object)
        with open(self.path+'dict.txt','r',encoding='UTF-8') as file_object:#这个真是魔鬼啊
            temp=file_object.read()
            lines=temp.split('\n')
            self.dic={}
            for line in lines:
                temp_word=[]
                temp_dictline=line.split('\t')
                for word in temp_dictline[1]:
                    temp_word.append(word)
                self.dic[temp_dictline[0]]=temp_word

    def decode(self,list_pinyin):
        len_pinyin=len(list_pinyin)
        #构造词网格.....
        for from_index in range(0,len_pinyin):
            for to_index in range(from_index,len_pinyin):
                get_word(list_pinyin[from_index:to_index])
    
    def get_word(self,list_pinyin_word):
        #没有语言模型文件的对应词典就先暴力解决吧....
        len_pinyin_word=len(list_pinyin_word)
        words=[]
        for i in range(0,len_pinyin_word):
            self.dic[list_pinyin_word[i]]

if __name__=='__main__':
    LM=Language_Model()