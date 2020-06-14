from pypinyin import pinyin,Style
import re
import os
def get_wav_list(wav_path,save_path):
    with open(wav_path,'r',encoding='UTF-8') as file_object,\
         open(save_path,'w',encoding='UTF-8') as file_object2:
        text_all=file_object.read()
        text_all=text_all.split('\n')
        for text in text_all:
            if text!='':
                text_list=text.split(' ')
                name=text_list[0]
                path='aidatatang200\\'+text_list[1]
                path=path.replace('/','\\')
                output=name+' '+path+'\n'
                file_object2.write(output)

def fix(wav_path,symbol_path,save_path):
        with open(wav_path,'r',encoding='UTF-8') as file_object,\
         open(symbol_path,'r',encoding='UTF-8') as file_object2,\
         open(save_path,'w',encoding='UTF-8') as file_object3:
            wav=file_object.readlines()
            symbol=file_object2.readlines()
            wav_list=[]
            for temp in wav:
                temp=temp.split(' ')
                wav_list.append(temp[0])
            for temp in symbol:
                temp=temp.split(' ')
                if temp[0] in wav_list:
                    file_object3.write(temp[0]+' '+' '.join(temp[1:]))


if __name__=='__main__':
    #get_wav_list('dataset\\数据集列表和中文拼音(aishell-1+primewords+aidatatang200)\\aidatatang_lst\\train.wav.lst','dataset\\aidata\\wav_train.txt')
    #get_wav_list('dataset\\数据集列表和中文拼音(aishell-1+primewords+aidatatang200)\\aidatatang_lst\\dev.wav.lst','dataset\\aidata\\wav_dev.txt')
    #get_wav_list('dataset\\数据集列表和中文拼音(aishell-1+primewords+aidatatang200)\\aidatatang_lst\\test.wav.lst','dataset\\aidata\\wav_test.txt')
    fix('dataset\\数据集列表和中文拼音(aishell-1+primewords+aidatatang200)\\aidatatang_lst\\train.wav.lst',
        'dataset\\数据集列表和中文拼音(aishell-1+primewords+aidatatang200)\\aidatatang_lst\\train.syllable.txt',
        'dataset\\aidata\\symbol_train.txt')

