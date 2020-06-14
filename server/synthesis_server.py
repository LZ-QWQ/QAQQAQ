import requests
def syn(text):#合成语音
    url = 'http://113.103.196.200:5000/synthesis'#113.103.196.224
    d = {'text': text}
    r = requests.post(url, data=d)
    #print(r.headers)
    music = r.content
    with open('temp.wav', 'wb') as file: #保存到本地的文件名
        file.write(music)
    print("合成完成")

def asr(path_filename):#识别音频
    url = 'http://113.103.196.200:5000/asr'
    files={'file':open(path_filename,'rb')}
    r=requests.post(url,files=files)
    if r.text!='error':
        temp=r.text.split('\n')
        print(temp[0])
        print(temp[1])
    else:
        print('音频太长,不能超过16秒')

if __name__=='__main__':
    asr('asr_test_file\\test.wav')
    syn('人生若只如初见，何事秋风悲画扇。等闲变却故人心，却道故人心易变。骊山语罢清宵半，泪雨霖铃终不怨。何如薄幸锦衣郎，比翼连枝当日愿。')
    #syn('比如')