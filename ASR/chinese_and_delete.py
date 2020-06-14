from glob import glob
import wav
import os 
def delete_():
    '''
    在dataset里面运行
    删除过长的音频的记录。。。。
    '''
    total_out_line=0
    filename_list=glob(os.path.join('datalist','*','wav_test.txt'),recursive=True)
    filename_list+=glob(os.path.join('datalist','*','wav_dev.txt'),recursive=True)
    print(filename_list)
    for filename in filename_list:
        temp=filename.split('\\')
        temp[-1]='symbol'+temp[-1][3:]
        symbol_name='\\'.join(temp)
        print(symbol_name)
        with open(filename,'r',encoding='UTF-8') as file_object,\
            open(symbol_name,'r',encoding='UTF-8') as f:
            temp_list=file_object.readlines()
            temp_list2=f.readlines()
        
        with open(filename,'w',encoding='UTF-8') as file_object, \
            open(symbol_name,'w',encoding='UTF-8') as f:
            for temp,temp2 in zip(temp_list,temp_list2):
                wav_filename=temp.split(' ')[1][0:-1]#去换行符，就笨一点吧
                wavsignal,framerate,wavetime = wav.read_wav_data(os.path.join('dataset',wav_filename))
                if(framerate != 16000):
                    raise ValueError('[Error] ASRT currently only supports wav audio files with a sampling rate of 16000 Hz, but this audio is ' + str(fs) + ' Hz. ')  
                time_length=len(wavsignal[0])/framerate*1000
                if time_length>=500 and time_length<16025: #我也不知道我以前为啥这样设置？？
                    file_object.write(temp)
                    f.write(temp2)
                    continue
                total_out_line+=1
                print(temp)
    print(total_out_line)

def get_chinese():
    filename_list=glob(os.path.join('datalist','*','symbol_train.txt'),recursive=True)
    filename_list+=glob(os.path.join('datalist','*','symbol_test.txt'),recursive=True)
    filename_list+=glob(os.path.join('datalist','*','symbol_dev.txt'),recursive=True)
if __name__=='__main__':
    delete_()