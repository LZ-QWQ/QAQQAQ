from pypinyin import pinyin,Style
import re
import os
import json
import wave

def prime2pinyin(path_filename,save_path_symbol,save_path_wav):
    with open(path_filename,'r',encoding='UTF-8') as file_object,\
        open(save_path_symbol,'w',encoding='UTF-8') as file_object2,\
        open(save_path_wav,'w',encoding='UTF-8') as file_object3:
        temp_all=json.load(file_object)
        for temp in temp_all: 
                text=temp['text']
                text=text.replace(' ','')
                file=temp['file']
                path='primewords_md_2018_set1\\audio_files\\'+file[0]+'\\'+file[0:2]+'\\'+file
                with wave.open('dataset\\'+path,mode='rb') as wav_object:
                    framerate=wav_object.getframerate()#采样频率
                    num_frame=wav_object.getnframes()#音频总帧数
                    times=float(num_frame)/framerate
                    if times<16.025:#！！！
                        name=file[0:-4]
                        pinyin_out=pinyin(text[0:],Style.TONE3)
                        len_pinyin=len(pinyin_out)
                        res=[]
                        for i in range(0,len_pinyin):
                            if re.search('\d',pinyin_out[i][0]):pass
                            else: 
                                pinyin_out[i][0]+='5'
                            res.append(pinyin_out[i][0])
                        res_str=' '.join(res)
                        file_object2.write(name+' '+res_str+'\n')
                        file_object3.write(name+' '+path+'\n')
                    else:
                        continue

if __name__=='__main__':
    path_filename='dataset\\primewords_md_2018_set1\\set1_transcript.json'
    save_path_symbol='dataset\\primewords_md_2018_set1\\symbol_train.txt'
    save_path_wav='dataset\\primewords_md_2018_set1\\wav_train.txt'
    prime2pinyin(path_filename,save_path_symbol,save_path_wav)