# -*- coding: utf-8 -*-
# @Time    : 2019/4/3 16:48
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : tmodel.py
# @Software: PyCharm
from tkinter import*
import tkinter.font as tkFont
import pandas as pd
import matplotlib as mpl
import numpy as np
from nltk.probability import FreqDist
import time
import requests
from jiebaSegment import *#
from sentenceSimilarity import SentenceSimilarity#
import pyaudio
import wave
from pyaudio import PyAudio,paInt16
framerate=16000
NUM_SAMPLES=2000
channels=1
sampwidth=2
TIME=5
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # enable chinese

# 设置外部词
seg = Seg()
seg.load_userdict('../userdict\\userdict.txt')


def play():
    CHUNK = 2014
    # 从目录中读取语音
    wf = wave.open('./hecheng.wav', 'rb')
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
def syn(text):#合成语音
    url = 'http://113.103.196.200:5000/synthesis'#
    d = {'text': text}
    r = requests.post(url, data=d)
    #print(r.headers)
    music = r.content
    with open('hecheng.wav', 'wb') as file: #保存到本地的文件名
        file.write(music)

def read_corpus1():
    qList = []
    # 问题的关键词列表
    qList_kw = []
    aList = []
    data = pd.read_csv('../data/corpus1/faq/qa_.csv', header=None)
    data_ls = np.array(data).tolist()
    for t in data_ls:
        qList.append(t[0])
        qList_kw.append(seg.cut(t[0]))
        aList.append(t[1])
    return qList_kw, qList, aList


def read_corpus2():
    qList = []
    # 问题的关键词列表
    qList_kw = []
    aList = []
    with open('../data/corpus1/chat/chat-small2.txt', 'r', encoding='utf-8') as f2:
        for i in f2:
            t = i.split('\t')
            s1 = ''.join(t[0].split(' '))
            s2 = ''.join(t[1].strip('\n'))
            qList.append(s1)
            qList_kw.append(seg.cut(s1))
            aList.append(s2)

    return qList_kw, qList, aList


def plot_words(wordList):
    fDist = FreqDist(wordList)
    #print(fDist.most_common())
    print("单词总数: ",fDist.N())
    print("不同单词数: ",fDist.B())
    fDist.plot(10)


def invert_idxTable(qList_kw):  # 定一个一个简单的倒排表
    invertTable = {}
    for idx, tmpLst in enumerate(qList_kw):
        for kw in tmpLst:
            if kw in invertTable.keys():
                invertTable[kw].append(idx)
            else:
                invertTable[kw] = [idx]
    return invertTable


def filter_questionByInvertTab(inputQuestionKW, questionList, answerList, invertTable):
    idxLst = []
    questions = []
    answers = []
    for kw in inputQuestionKW:
        if kw in invertTable.keys():
            idxLst.extend(invertTable[kw])
    idxSet = set(idxLst)
    for idx in idxSet:
        questions.append(questionList[idx])
        answers.append(answerList[idx])
    return questions, answers


def main(question, top_k, task='faq'):
    # 读取数据
    if task == 'chat':
        qList_kw, questionList, answerList = read_corpus2()
    else:
        qList_kw, questionList, answerList = read_corpus1()

    """简单的倒排索引"""
    # 计算倒排表
    invertTable = invert_idxTable(qList_kw)
    inputQuestionKW = seg.cut(question)

    # 利用关键词匹配得到与原来相似的问题集合
    questionList_s, answerList_s = filter_questionByInvertTab(inputQuestionKW, questionList, answerList,
                                                              invertTable)
    # 初始化模型
    ss = SentenceSimilarity(seg)
    ss.set_sentences(questionList_s)
    ss.TfidfModel()  # tfidf模型
    # ss.LsiModel()         # lsi模型
    # ss.LdaModel()         # lda模型
    question_k = ss.similarity_k(question, top_k)
    return question_k, questionList_s, answerList_s

if __name__ == '__main__':
    # 设置外部词
   
    def printInfo(event):    
        seg = Seg()
        seg.load_userdict('../userdict/userdict.txt')
        # 读取数据
        List_kw, questionList, answerList = read_corpus1()
        # 初始化模型
        ss = SentenceSimilarity(seg)
        ss.set_sentences(questionList)
        ss.TfidfModel()         # tfidf模型
        # ss.LsiModel()         # lsi模型
        # ss.LdaModel()         # lda模型
        text2.delete(1.0, END)
        question=(text1.get('1.0',END))
        
        #if question == 'q':
            #break
        time1 = time.time()
        question_k = ss.similarity_k(question, 5)
        text2.insert("insert","： {}".format(answerList[question_k[0][0]]))
        #print("： {}".format(answerList[question_k[0][0]]))
        #for idx, score in zip(*question_k):
           # print("same questions： {},                score： {}".format(questionList[idx], score))
        #time2 = time.time()
        #cost = time2 - time1
        #print('Time cost: {} s'.format(cost))
        #entry2.insert(10,question)
        #清空entry2控件
        text1.delete(1.0, END)
        syn("： {}".format(answerList[question_k[0][0]]))
    def printInfo2(event):
        play()
    myWindow = Tk()
    myWindow.title('Python GUI Learning')
    ft = tkFont.Font(size=20, slant=tkFont.ITALIC)
    ft1 = tkFont.Font(size=10, slant=tkFont.ITALIC)
    #myButton = Button(myWindow,text = "提问")
    Label1 = Label(myWindow,text = "按F1得答案",width=20,height=8,font=ft)
    Label1.grid(row=31, column=0)
    Label2 = Label(myWindow,text = "按F2得播放语音",width=20,height=8,font=ft)
    Label2.grid(row=31, column=1)

    text2=Text(myWindow,width=100,height=18,font=ft)
    
    text2.grid(row=30, column=1)
    text1=Text(myWindow,width=100,height=5,font=ft)
    text1.grid(row=29, column=1)
    Label(myWindow, text="欢迎使用西电智能系统:").grid(row=29)
    Label(myWindow, text="answer").grid(row=30)
    #entry1=Entry(myWindow,width=200)
    #entry2=Entry(myWindow,width=200)
    #entry1.grid(row=29, column=1)
    #entry2.grid(row=1, column=1)
        #清理entry2
        
    myWindow.bind('<F1>', printInfo)
    myWindow.bind('<F2>', printInfo2)

    myWindow.mainloop()
    
    
        #question = input("欢迎使用西电问答系统: ")
        #if question == 'q':
           # break
        #time1 = time.time()
        #question_k = ss.similarity_k(question, 5)
        #print("： {}".format(answerList[question_k[0][0]]))
        #for idx, score in zip(*question_k):
         #   print("same questions： {},                score： {}".format(questionList[idx], score))
        #time2 = time.time()
        #cost = time2 - time1
        #print('Time cost: {} s'.format(cost))








