import wave
import numpy as np
import matplotlib.pyplot as plt

import scipy.io.wavfile as wav #python_speech_feature doc上的

from scipy.fftpack import fft 
import python_speech_features as psf

from LZ_Error import LZ_Error

def read_wav_data(filename):
    '''
    读取wav文件，返回声音信号数据？？这叫啥（声音信号的时域谱矩阵）
    '''
    with wave.open(filename,mode='rb') as wav_object:
        num_channels=wav_object.getnchannels()#声道数
        sample_width=wav_object.getsampwidth()#采样字节长度
        framerate=wav_object.getframerate()#采样频率
        num_frame=wav_object.getnframes()#音频总帧数
        str_data=wav_object.readframes(num_frame)#读取全部音频 (bytes)
        #print(str_data)
        #print(len(str_data))
        #print(framerate)
        wave_data = np.fromstring(str_data, dtype = np.short)#转类型~
        wave_data.shape=-1,num_channels
        wave_data=wave_data.T#为什么一定要转置！魔鬼！

        wavtime=np.arange(0,num_frame)*(1.0/framerate)#时间哦！画图的，不过我为什么要画图？？
        #print(wavtime.shape)
    return wave_data,framerate,wavtime

def wave_show(wave_data,wavtime):
    plt.title("xxx.wav's Frames")
    #plt.subplot(211)
    plt.plot(wavtime, wave_data[0],'b')
    #plt.subplot(212)
    #plt.plot(wavtime, wave_data[1],'r')
    plt.show()

def get_spectrogram(wavsignal, framerate):#就是所谓的语谱图 log+spectrogram
    '''
    提取语谱图
    '''
    x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1) ) # 汉明窗
    if(framerate != 16000):
        raise ValueError('[Error] ASRT currently only supports wav audio files with a sampling rate of 16000 Hz, but this audio is ' + str(fs) + ' Hz. ')
    
    # wav波形 加时间窗以及时移10ms
    time_window = 25 # 单位ms
    window_length = framerate / 1000 * time_window # 计算窗长度的公式，目前全部为400固定值
    
    wav_arr = np.array(wavsignal)
    #print(wav_arr.shape)
    #wav_length = len(wavsignal[0])
    wav_length = wav_arr.shape[1]
    
    time_length=len(wavsignal[0])/framerate*1000
    #if time_length<500:#本来是为了避免短于25ms的，就直接这样吧
    #    raise LZ_Error('音频时长过短，时长为：'+str(time_length)+'ms')
    #elif time_length>=16025:
    #    raise LZ_Error('音频时长过长，时长为：'+str(time_length)+'ms')

    range0_end = int(time_length - time_window) // 10 + 1 # 计算循环终止的位置，也就是最终生成的窗数
    data_input = np.zeros((range0_end, int(window_length // 2)),np.float64) # 用于存放最终的频率特征数据
    data_line = np.zeros((1, int(window_length)), dtype = np.float64)
    
    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400
        
        data_line = wav_arr[0, p_start:p_end]
        
        data_line = data_line * w # 加窗
        
        data_line = np.abs(fft(data_line))#/wav_length 去掉  后面有对数，归一化吗？有空再研究

        data_input[i]=data_line[0: int(window_length // 2)] # 设置为400除以2的值（即200）是取一半数据，因为是对称的
        
    data_input = np.log(data_input + 1)
    data_input=np.array(data_input)
    return data_input

def GetMFCC(wave_data,framerate):
    QAQ_feat_mfcc=psf.mfcc(wave_data,framerate)#一、二阶微分据说效果好？！
    d_feat_mfcc=psf.delta(QAQ_feat_mfcc,2)
    dd_feat_mfcc=psf.delta(QAQ_feat_mfcc,2)
    feat_mfcc=np.hstack((QAQ_feat_mfcc,d_feat_mfcc,dd_feat_mfcc))#没懂跟column_stack的区别是啥？！
    return feat_mfcc

def spectrogram_show(feat):
    feat=feat.T
    #plt.figure()
    #plt.subplot(111)
    plt.imshow(feat)
    plt.colorbar(shrink=0.5)  
    plt.show() 

if(__name__=='__main__'):
    #wave_data,framerate,wavtime=read_wav_data('dataset\\data_thchs30\\data\\A2_0.wav')
    wave_data,framerate,wavtime=read_wav_data('dataset\\aidatatang200\\G0002\\session01\\T0055G0002S0002.wav')
    #wave_show(wave_data,wavtime)
    feat=get_spectrogram(wave_data,framerate)
    spectrogram_show(feat)
    pass