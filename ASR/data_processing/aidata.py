from pypinyin import pinyin,Style
import re
import os
#有毒！
def aidata2pinyin(text):
            pinyin_out=pinyin(text[0:],Style.TONE3)
            len_pinyin=len(pinyin_out)
            res=[]
            for i in range(0,len_pinyin):
                if re.search('\d',pinyin_out[i][0]):pass
                else: 
                    pinyin_out[i][0]+='5'
                res.append(pinyin_out[i][0])
            res_str=' '.join(res)
            return res_str

def get_wav_list(path,save_path):
    with open(save_path,'w',encoding='UTF-8') as file_object:
        for root,dirs,files in os.walk(path):
            for filename in files:
                if filename.endswith('.wav'):
                    res=os.path.join(root,filename)
                    res=res.replace('dataset\\','')#其实
                    name=filename[0:-4]#这两个办法去除这几个字符串都挺碰巧的
                    file_object.write(name+' '+res+'\n')

def get_symbol_list(path,save_path):
    with open(save_path,'w',encoding='UTF-8') as file_object:
        for root,dirs,files in os.walk(path):
            for filename in files:
                if filename.endswith('.txt'):
                    with open(root+'\\'+filename,'r',encoding='UTF-8') as file_object2:
                        text=file_object2.readlines()#貌似那该死的有两个空行？？
                        text=text[0]
                        text=text.replace('\n','')#貌似有换行
                        text=text.replace('。','')
                        text=text.replace('，','')
                        text=text.replace('？','')
                        text=text.replace('！','')
                        text=text.replace(',','')
                        text=text.replace('.','')
                        name=filename[0:-4]#这两个办法去除这几个字符串都挺碰巧的
                        pinyin=aidata2pinyin(text)
                    file_object.write(name+' '+pinyin+'\n')

if __name__=='__main__':
    #get_wav_list('dataset\\aidatatang200','dataset\\wav_train.txt')
    get_symbol_list('dataset\\aidatatang200','dataset\\symbol_train.txt')