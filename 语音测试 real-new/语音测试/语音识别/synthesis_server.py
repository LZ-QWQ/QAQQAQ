import requests
def syn(text):#合成语音
    url = 'http://113.103.196.224:5000/synthesis'#113.103.196.224
    d = {'text': text}
    r = requests.post(url, data=d)
    #print(r.headers)
    music = r.content
    with open('hecheng.wav', 'wb') as file: #保存到本地的文件名
        file.write(music)

def asr(path_filename):#识别音频
    url = 'http://113.103.196.224:5000/asr'
    files={'file':open(path_filename,'rb')}
    r=requests.post(url,files=files)
    if r.text!='error':
        temp=r.text.split('\n')
        print(temp[0])
        print(temp[1])
    else:
        print('音频太长,不能超过16秒')

if __name__=='__main__':
    asr('D:\\语音测试\\语音识别\\01.wav')
    syn('杜渡是大帅哥！')
