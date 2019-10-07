import keras.backend as K
from keras.layers import Lambda,Conv2D,Input,Dropout,MaxPooling2D,Reshape,Dense,Activation
from keras.models import Model,load_model
from keras.optimizers import Adam,Adadelta,SGD
from keras.callbacks import Callback
from keras.utils import multi_gpu_model,plot_model
import tensorflow as tf

import platform as plat
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import difflib

from readdata import DataSpeech 


class SpeechModel():
    def __init__(self, relpath):#relpath 数据集相对路径
        self.OUTPUT_SIZE=1365;#很尴尬，，最后一个空白
        self.STRING_LENGTH=64;
        self.AUDIO_LENGTH=1600;
        self.AUDIO_FEATURE_LENGTH=200;#MFCC39 nl的是200

    
    
        system_type = plat.system()
        abspath_file = os.path.abspath(os.path.dirname(__file__))
     
        self.relpath=relpath
        self.slash = ''
        if(system_type == 'Windows'):
           self.slash = '\\' # 反斜杠
        elif(system_type == 'Linux'):
           self.slash = '/' # 正斜杠
        else:
           print('[Message] Unknown System\n')
           self.slash = '/' # 正斜杠
        
        self.relpath=relpath
        self.datapath = abspath_file + self.slash + relpath + self.slash#还不知道咋用
        self.save_path='model_save'+self.slash#保存模型的路径

        self.CreateModel()#产生self.model_data和self.model_ctc

    def CreateModel(self):
        
        input_data=Input(shape=(self.AUDIO_LENGTH,self.AUDIO_FEATURE_LENGTH,1),dtype='float32',name='input_data')
        #这个 1 是为了后面的conv2d
        layer_1=Dropout(0.1)(input_data)
        
        layer_1=Conv2D(32,(3,3),use_bias=True,activation='relu',padding='same',
                       kernel_initializer='he_normal')(layer_1)
        #layer_1=Dropout(0.05)(layer_1)
        layer_2=Conv2D(32,(3,3),use_bias=True,activation='relu',padding='same',
                       kernel_initializer='he_normal')(layer_1)
        
        layer_3=MaxPooling2D(pool_size=(2,2),strides=None,padding='valid')(layer_2)
        layer_3=Dropout(0.05)(layer_3)

        layer_4=Conv2D(64,(3,3),use_bias=True,activation='relu',padding='same',
                       kernel_initializer='he_normal')(layer_3)
        #layer_4=Dropout(0.1)(layer_4)
        layer_5=Conv2D(64,(3,3),use_bias=True,activation='relu',padding='same',
                       kernel_initializer='he_normal')(layer_4)

        layer_6=MaxPooling2D(pool_size=(2,2),strides=None,padding='valid')(layer_5)
        layer_6=Dropout(0.1)(layer_6)

        layer_7=Conv2D(64,(3,3),use_bias=True,activation='relu',padding='same',
                       kernel_initializer='he_normal')(layer_6)
        #layer_7=Dropout(0.15)(layer_7)
        layer_8=Conv2D(64,(3,3),use_bias=True,activation='relu',padding='same',
                       kernel_initializer='he_normal')(layer_7)

        layer_9=MaxPooling2D(pool_size=(2,2),strides=None,padding='valid')(layer_8)
        layer_9=Dropout(0.15)(layer_9)

        layer_10=Conv2D(128,(3,3),use_bias=True,activation='relu',padding='same',
                       kernel_initializer='he_normal')(layer_9)
        #layer_10=Dropout(0.2)(layer_10)
        layer_11=Conv2D(128,(3,3),use_bias=True,activation='relu',padding='same',
                       kernel_initializer='he_normal')(layer_10)

        layer_12=MaxPooling2D(pool_size=(1,1),strides=None,padding='valid')(layer_11)#就是搞笑的不能再除了
        layer_12=Dropout(0.2)(layer_12)

        layer_13=Conv2D(128,(3,3),use_bias=True,activation='relu',padding='same',
                       kernel_initializer='he_normal')(layer_12)
        #layer_13=Dropout(0.25)(layer_13)
        layer_14=Conv2D(128,(3,3),use_bias=True,activation='relu',padding='same',
                       kernel_initializer='he_normal')(layer_13)

        layer_15=MaxPooling2D(pool_size=(1,1),strides=None,padding='valid')(layer_14)#就是搞笑的不能再除了
        layer_15=Dropout(0.25)(layer_15)

        layer_16=Reshape((200,3200))(layer_15)#这个200很关键
        layer_17=Dense(2048,activation='relu',use_bias=True,kernel_initializer='he_normal')(layer_16)
        layer_17=Dropout(0.35)(layer_17)
        #layer_18=Dense(2048,activation='relu',use_bias=True,kernel_initializer='he_normal')(layer_17)
        

        layer_18=Dense(self.OUTPUT_SIZE,use_bias=True,kernel_initializer='he_normal')(layer_17)
        y_pred=Activation('softmax',name='softmax')(layer_18)
        
        self.model_data=Model(input_data,y_pred)
        self.model_data.summary()

        label=Input(shape=[self.STRING_LENGTH],dtype='float32',name='label')#这个要么用[]要么要(1,)要不报错哎
        input_length=Input(shape=(1,),dtype='int64',name='input_length')#(1)可能有问题把
        label_length=Input(shape=(1,),dtype='int64',name='label_length')        
        
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out=Lambda(self.ctc_loss_func,output_shape=(1,),
                    name='ctc')([y_pred,label,input_length,label_length])

        with tf.device('/cpu:0'):#好像这里没有用啊？是默认cpu建立了吗
            self.model_ctc=Model([input_data,label,input_length,label_length],loss_out)

        self.model_ctc.summary()

        #self.parallel_model_ctc = multi_gpu_model(self.model_ctc,gpus=2, cpu_relocation=True)
        #cpu_relocation=True的话会报错，，为什么。。。查了下没查到
        self.parallel_model_ctc=multi_gpu_model(self.model_ctc,gpus=2)
        
        #opt = Adadelta(lr = 0.01, rho = 0.95, epsilon = 1e-06)
        opt=Adam(lr=0.00005)#Adam默认参数传说中的
        #opt=SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5) 
        
        #self.model_ctc.compile(loss={'ctc' : lambda y_true,y_pred:y_pred},
        #                       optimizer=opt,metrics = ['accuracy'])
        self.parallel_model_ctc.compile(loss={'ctc' : lambda y_true,y_pred:y_pred},
                               optimizer=opt,metrics = ['accuracy'])
        #这个accuracy好像没用啊.....这个输出是ctc,,,,,,

        # captures output of softmax so we can decode the output during visualization
        #这个留着以后改进回调函数用
        test_func = K.function([input_data], [y_pred])

    def ctc_loss_func(self,args):
        y_pred,label,input_length,label_length=args
        # the 2 is critical here since the first couple outputs of the RNN
        # tend to be garbage:
        #y_pred = y_pred[:, 2:, :] 然后博士保留了这个垃圾hhh
        #y_pred = y_pred[:, :, :]

        return K.ctc_batch_cost(label,y_pred,input_length,label_length)

    def TrainModel(self,filename,batch_size=32,epochs=50,save_epoch=1):
        '''
        filename:保存的文件名，路径、后缀已准备好，如model_lz
        '''
        speech_datas=DataSpeech(self.relpath,'train')
        #speech_validation=DataSpeech(self.relpath,'test')#验证数据
        data_nums=speech_datas.DataNum_Total
        #validation_nums=speech_validation.DataNum_Total
        yield_datas=speech_datas.nl_speechmodel_generator(32,self.AUDIO_LENGTH,self.STRING_LENGTH)
        #yield_validation=speech_validation.nl_speechmodel_generator(8,self.AUDIO_LENGTH,self.STRING_LENGTH)
        for epoch in range(0,epochs//save_epoch):#这个地方感觉可以改进一下，要不换个办法？？
            print("[提示QAQ]已经训练%d轮次，共%d轮(一轮数据量应为（500*batch_size=%d）)"
                  %(epoch*save_epoch,epochs,500*batch_size))
            try:
                hist=LossHistory()
                self.parallel_model_ctc.fit_generator(generator=yield_datas,
                                              steps_per_epoch=500,
                                              epochs=save_epoch,
                                              verbose=1,
                                              #validation_data=yield_validation,
                                              #validation_steps=50,
                                              callbacks=[hist]
                                              )
            except StopIteration:
                print("[错误QAQ]貌似生成的数据格式有点问题？？")#天知道触发吗
                break

            hist.plot()
            hist.save('lz_new.json')
            self.SaveModel(filename)
    
        
        epoch_residue=epochs%save_epoch#防止不整除
        if  epoch_residue!=0:
            print("[提示QAQ]已经训练%d轮次，共%d轮(一轮数据量应为（500*batch_size=%d）)"
                  %(epochs-epoch_residue,epochs,500*batch_size))
            try:
                hist=LossHistory()
                self.parallel_model_ctc.fit_generator(generator=yield_datas,
                                              steps_per_epoch=500,
                                              epochs=epoch_residue,
                                              verbose=1,
                                              #validation_data=yield_validation,
                                              #validation_steps=50,
                                              callbacks=[hist]
                                              )
                hist.plot()
                hist.save('lz_new.json')
                self.SaveModel(filename)
                #其实可以用ModelCheckpoint来保存。。。。。。写都写好了额
            except StopIteration:
                print("[错误QAQ]貌似生成的数据格式有点问题？？")#天知道触发吗
                
        print("[提示QAQ]已经训练%d轮次，共%d轮(一轮数据量应为（500*batch_size=%d）)"
                  %(epochs,epochs,500*batch_size))

    def SaveModel(self,filename):
        '''
        filename:保存的文件名，路径、后缀已准备好，如model_lz
        '''
        self.model_ctc.save(self.save_path+filename+'_ctc.h5')
        self.model_data.save(self.save_path+filename+'_data.h5')
        self.model_ctc.save_weights(self.save_path+filename+'_weights_ctc.h5')
        self.model_data.save_weights(self.save_path+filename+'_weights_data.h5')

    def LoadModel(self,filename):
        '''
        filename:保存的文件名，路径、后缀已准备好，如model_lz
        '''
        self.model_ctc.load_weights(self.save_path+filename+'_weights_ctc.h5')
        self.model_data.load_weights(self.save_path+filename+'_weights_data.h5')

    def PredModel(self,filename):
        '''
        filename:保存的文件名，路径、后缀已准备好，如model_lz
        '''
        path='model_save'+self.slash
        self.model_data.load_weights(path+filename+'_weights_data.h5')
        speech_datas=DataSpeech(self.relpath,'train')
        data_input,data_output=speech_datas.GetData(0)
        X=np.zeros((1,1600,200,1),dtype=np.float64)
        X[0,0:len(data_input)]=data_input;
        y_pre=self.model_data.predict(X)
        input_length = np.zeros((1),dtype = np.int32)
        input_length[0]=(min(data_input.shape[0] // 8 + 
                                    data_input.shape[0] % 8,200))
        r=K.ctc_decode(y_pre,input_length)
        r1 = r[0][0]
        n=K.eval(r1)
        m=K.get_value(r[1])
        print(n)#get_value好像是把张量变数组
        #print(n.shape)
        #print(type(n))
        print(m)
        pinyin_r=speech_datas.num2symbol(data_output)
        total_pinyin=len(pinyin_r)
        pinyin=speech_datas.num2symbol(n[0])
        WER=GetEditDistance_pinyin(pinyin,pinyin_r)/total_pinyin
        print(pinyin)
        print('WER:'+str(WER))

    def VisualModel(self,filename):
        '''
        filename:保存的文件名，路径、后缀已准备好，如model_lz
        '''
        path='model_image'+self.slash
        plot_model(self.model_ctc,to_file=path+'model_ctc.png',show_shapes=True)
        plot_model(self.model_data,to_file=path+'model_data.png',show_shapes=True)

class LossHistory(Callback):
    def __init__(self):
        super(LossHistory,self).__init__()
        self.loss = {'batch':[], 'epoch':[]}
        self.acc = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

        system_type = plat.system()
        self.slash = ''
        if(system_type == 'Windows'):
           self.slash = '\\' # 反斜杠
        elif(system_type == 'Linux'):
           self.slash = '/' # 正斜杠
        else:
           print('[Message] Unknown System\n')
           self.slash = '/' # 正斜杠

    def on_batch_end(self, batch, logs={}):
        self.loss['batch'].append(float(logs.get('loss')))
        self.acc['batch'].append(float(logs.get('accuracy')))
        #self.val_loss['batch'].append(logs.get('val_loss'))
        #self.val_acc['batch'].append(logs.get('val_acc'))
        #好像这不对哦，fit_generator的参数介绍是下面这么写的，可测一下，，，
        #on which to evaluate
        #the loss and any model metrics at the end of each epoch.

    def on_epoch_end(self, batch, logs={}):
        self.loss['epoch'].append(float(logs.get('loss')))
        self.acc['epoch'].append(float(logs.get('accuracy')))
        #self.val_loss['epoch'].append(logs.get('val_loss'))
        #self.val_acc['epoch'].append(logs.get('val_acc'))

    def save(self,filename):
        '''保存的文件名称'''
        #这个地方还是要换行保存才行！
        path='loss_acc_save'+self.slash
        with open(path+filename,mode='a') as file_object:
            #json.dump({'loss':self.loss,'acc':self.acc},file_object,indent=2)
            temp=json.dumps({'loss':self.loss['epoch'],'acc':self.acc['epoch']})
            file_object.write(temp+'\n')

    def plot(self,loss_type='batch'):#现在epoch都是1以后改进
        '''loss_type:batch,epoch'''
        iters = range(len(self.loss[loss_type]))
        fig=plt.figure()
        # acc
        plt.plot(iters, self.acc[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.loss[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.pause(5)
        plt.close(fig)

def GetEditDistance_pinyin(y_pred,label):#莱文斯坦距离、Levenshtein距离，动态规划实现，拼音List
    '''
    计算两个拼音列表之间的编辑距离
    采用 构造矩阵动态规划的方式
    '''
    len_pred=len(y_pred)
    len_label=len(label)
    D=np.zeros((len_label+1,len_pred+1),dtype=np.int16)
    for i in range(0,len_label+1):
        for j in range(0,len_pred+1):
            if i==0:
                D[0][j]=j
            elif j==0:
                D[i][0]=i

    for i in range(1,len_label+1):
        for j in range(1,len_pred+1):
            if label[i-1]==y_pred[j-1]:#列表索引比矩阵少1
                D[i][j]=D[i-1][j-1]
            else:
                sub=D[i-1][j-1]+1
                ins=D[i][j-1]+1
                delete=D[i-1][j]+1
                D[i][j]=min(sub,delete,ins)

    print(D[len_label][len_pred])
    return D[len_label][len_pred]

def GetEditDistance_str(str_pred,str_label):#difflib实现 汉字字符串
    leven_cost = 0
    str_pred=str_pred.replace(' ','')#isjunk参数有毒
    str_label=str_label.replace(' ','')
    s = difflib.SequenceMatcher(None, str_pred, str_label)
    for (tag, i1, i2, j1, j2) in s.get_opcodes():
        if tag == 'replace':
            leven_cost += max(i2-i1, j2-j1)
        elif tag == 'insert':
            leven_cost += (j2-j1)
        elif tag == 'delete':
            leven_cost += (i2-i1)
    print(leven_cost)
    return leven_cost

if(__name__ == '__main__'):
    #S_M=SpeechModel('dataset')
    #S_M.LoadModel('lz_new')
    #S_M.TrainModel('lz_new',epochs=200,save_epoch=1)
    #S_M.PredModel('lz_new')
    GetEditDistance_pinyin(['ni3','hao3'],['bu4','hao3','a1'])
    GetEditDistance_str('绿 是 阳春 烟 警 大 快 文章 的 底色 四月 的 林 栾 更 是 率 的 先 活 秀媚 失意 盎然',
                        '绿 是 阳春 烟 景 大块 文章 的 底色 四月 的 林 峦 更是 绿 得 鲜活 秀媚 诗意 盎然')