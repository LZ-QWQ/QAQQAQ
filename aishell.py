from pypinyin import pinyin,Style
import re
import os

def aishell2pinyin(path_filename,save_path):
    with open(path_filename,'r',encoding='UTF-8') as file_object,\
        open(save_path,'w',encoding='UTF-8') as file_object2:
        temp_all=file_object.read()
        temp_list=temp_all.split('\n')
        for temp in temp_list:
            if temp!='':
                text=temp.split(' ')
                pinyin_out=pinyin(text[1:],Style.TONE3)
            else : 
                print('嘿嘿')
                break
            len_pinyin=len(pinyin_out)
            res=[]
            for i in range(0,len_pinyin):
                if re.search('\d',pinyin_out[i][0]):pass
                else: 
                    pinyin_out[i][0]+='5'
                res.append(pinyin_out[i][0])
            res_str=' '.join(res)
            file_object2.write(text[0]+' '+res_str+'\n')

def get_wav_list(path,save_path):
    with open(save_path,'w',encoding='UTF-8') as file_object:
        for root,dirs,files in os.walk(path):
            for filename in files:
                if filename.endswith('.wav'):
                    res=os.path.join(root,filename)
                    res=res.replace('dataset\\','')#其实
                    name=filename.rstrip('.wav')#这两个办法去除这几个字符串都挺碰巧的
                    file_object.write(name+' '+res+'\n')

def get_symbol_list(path_pinyin,path_wav,save_path):
    with open(path_pinyin,'r',encoding='UTF-8') as file_object,\
        open(path_wav,'w',encoding='UTF-8') as file_object2,\
        open(save_path,'w',encoding='UTF-8') as file_object3:
        pass

if __name__=='__main__':
    path_filename='dataset\\data_aishell\\transcript\\aishell_transcript_v0.8.txt'
    save_path='dataset\\data_aishell\\transcript\\aishell_pinyin.txt'
    #aishell2pinyin(path_filename,save_path)
    get_wav_list('dataset\\data_aishell\\wav\\train','dataset\\data_aishell\\wav\\wav_train.txt')