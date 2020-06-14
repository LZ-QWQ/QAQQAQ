import wave
from pyaudio import PyAudio,paInt16
from tkinter import*
import tkinter.font as tkFont
import requests
import time
import pyaudio

framerate=16000
NUM_SAMPLES=2000
channels=1
sampwidth=2
TIME=3
def syn(text):#合成语音
    url = 'http://113.103.196.200:5000/synthesis2'#113.103.196.224
    d = {'text': text}
    r = requests.post(url, data=d)
    #print(r.headers)
    music = r.content
    with open('hecheng.wav', 'wb') as file: #保存到本地的文件名
        file.write(music)

#def asr(path_filename):#识别音频
#    url = 'http://113.103.196.224:5000/asr'
#    files={'file':open(path_filename,'rb')}
#    r=requests.post(url,files=files)
#    if r.text!='error':
#        temp=r.text.split('\n')
#        #print(temp[0])
#        #print(temp[1])
#    else:
#        #print('音频太长,不能超过16秒')
def save_wave_file(filename,data):
    '''save the date to the wavfile'''
    wf=wave.open(filename,'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b"".join(data))
    wf.close()

def my_record():
    pa=PyAudio()
    stream=pa.open(format = paInt16,channels=1,
                   rate=framerate,input=True,
                   frames_per_buffer=NUM_SAMPLES)
    my_buf=[]
    count=0
    while count<TIME*8:#控制录音时间
        string_audio_data = stream.read(NUM_SAMPLES)
        my_buf.append(string_audio_data)
        count+=1
        print('.')
    save_wave_file('01.wav',my_buf)
    stream.close()

chunk=2014
def play():
    CHUNK = 1024
    # 从目录中读取语音
    wf = wave.open('01.wav', 'rb')
    # read data
    data = wf.readframes(CHUNK)
    # 创建播放器
    p = pyaudio.PyAudio()

    # 获得语音文件的各个参数
    FORMAT = p.get_format_from_width(wf.getsampwidth())
    CHANNELS = wf.getnchannels()
    RATE = wf.getframerate()

    print('FORMAT: {} \nCHANNELS: {} \nRATE: {}'.format(FORMAT, CHANNELS, RATE))
    # 打开音频流， output=True表示音频输出
    stream = p.open(format=FORMAT,

                    channels=CHANNELS,
                    rate=RATE,
                    frames_per_buffer=CHUNK,
                    output=True)
    # play stream (3) 按照1024的块读取音频数据到音频流，并播放
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(CHUNK)
if __name__ == '__main__':
    def printInfo(event):
        my_record()
        print('end')
        url = 'http://113.103.196.200:5000/asr'
        files={'file':open("01.wav",'rb')}
        r=requests.post(url,files=files)
        if r.text!='error':
            temp=r.text.split('\n')
            text1.insert("insert",temp[1]+'    '+temp[0]+'\n')
        else:
            text1.insert("音频太长")
        syn(temp[0])
        print("666")
    def printInfo2(event):
        play()    
    myWindow = Tk()
    myWindow.title('Python GUI Learning')
    ft = tkFont.Font(size=20, slant=tkFont.ITALIC)
    Label1 = Label(myWindow,text = "按F1得开始录音",width=30,height=5,font=ft)
    Label2 = Label(myWindow,text = "按F2得开始播放",width=30,height=5,font=ft)
    Label1.grid(row=31, column=1)
    Label2.grid(row=32, column=1)
    text1=Text(myWindow,width=100,height=5,font=ft)
    text1.grid(row=29, column=1)
    Label(myWindow, text="欢迎使用西电语音识别系统:").grid(row=29)
    myWindow.bind('<F1>', printInfo)
    myWindow.bind('<F2>', printInfo2)
    myWindow.mainloop()


    

