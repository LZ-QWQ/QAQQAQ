import keras.backend as K
from keras.layers import Lambda,Conv2D,Input,Dropout,MaxPooling2D,Reshape,Dense,Activation
from keras.models import Model,load_model,clone_model
from keras.optimizers import Adam,Adadelta,SGD
from keras.callbacks import Callback
from keras.utils import multi_gpu_model,plot_model
import tensorflow as tf

import pickle
import platform as plat
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import difflib
import random

from readdata import DataSpeech 
from LZ_Error import LZ_Error

class SpeechModel():
    def __init__(self, relpath):#relpath 数据集相对路径
        self.OUTPUT_SIZE=1366;#很尴尬，，最后一个空白
        self.STRING_LENGTH=64;
        self.AUDIO_LENGTH=1600;
        self.AUDIO_FEATURE_LENGTH=200;#MFCC39 nl的是200

    
    
        system_type = plat.system()
        abspath_file = os.path.abspath(os.path.dirname(__file__))
     
        self.relpath=relpath
        
        self.save_path='model_save'#保存模型的路径

        self.CreateModel()#产生self.model_data和self.model_ctc
        self.model_data.summary()
        self.model_ctc.summary()
        
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
        

        label=Input(shape=[self.STRING_LENGTH],dtype='float32',name='label')#这个要么用[]要么要(1,)要不报错哎
        input_length=Input(shape=(1,),dtype='int64',name='input_length')#(1)可能有问题把
        label_length=Input(shape=(1,),dtype='int64',name='label_length')        
        
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out=Lambda(ctc_loss_func,output_shape=(1,),
                    name='ctc')([y_pred,label,input_length,label_length])
 
        self.model_data=Model(input_data,y_pred)
        self.model_ctc=Model([input_data,label,input_length,label_length],loss_out)
        #self.model_data.summary()
        #self.model_ctc.summary()
       

    def TrainModel(self,filename,batch_size=32,epochs=50,save_epoch=1):
        '''
        filename:保存的文件名，路径、后缀已准备好，如model_lz
        '''       
        self.epoch_all=0
        if os.path.exists(os.path.join(self.save_path,filename+'_epochs.pkl')):
            with open(os.path.join(self.save_path,filename+'_epochs.pkl'),'rb') as f:
                self.epoch_all=pickle.load(f)
            print('加载epochs数',self.epoch_all)
        else:
            print('未加载epochs数，设为',self.epoch_all)
      
        #这里来加载已存在的模型权重
        if os.path.exists(os.path.join(self.save_path,filename+'_weights_ctc_'+str(self.epoch_all)+'.h5')):
            self.model_ctc.load_weights(os.path.join(self.save_path,filename+'_weights_ctc_'+str(self.epoch_all)+'.h5'))
            
        lr=0.0001#https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
        lr=0.00005
        lr=0.00002
        lr=0.00001
        lr=0.000005
        lr=0.000004
        #lr=1e-6
        #opt=Adam(lr=lr,decay=lr/epochs)#Adam默认参数传说中的,衰减这样设置试试？
        opt=Adam(lr=lr)#上面这个衰减好像有点麻烦，，有点不对劲，要衰减要用callback去改比较好，，要命哦        
        #https://blog.csdn.net/zzc15806/article/details/79711114
        from keras.callbacks import LearningRateScheduler
 
        def scheduler(epoch):
            init_lr=lr
            new_lr=lr*0.5**(epoch//2)#试一下这个嘿嘿
            return new_lr
            
        reduce_lr = LearningRateScheduler(scheduler,verbose=1)#暂时先不用               
        def ctc_loss(y_true,y_pred):#定义一般的损失函数
            return y_pred
        self.model_ctc.compile(loss={'ctc' : ctc_loss},optimizer=opt,metrics = ['accuracy'])#这个accuracy好像没用啊.....这个输出是ctc,,,,,,
                
        speech_datas=DataSpeech(self.relpath,'train')
        #speech_validation=DataSpeech(self.relpath,'test')#验证数据
        data_nums=speech_datas.DataNum_Total
        #validation_nums=speech_validation.DataNum_Total
        yield_datas=speech_datas.speechmodel_generator(batch_size,self.AUDIO_LENGTH,self.STRING_LENGTH)
        #yield_validation=speech_validation.nl_speechmodel_generator(8,self.AUDIO_LENGTH,self.STRING_LENGTH)   

        print("[提示QAQ]一个epoch的数据量为%d"%data_nums)
        try:
            hist=LossHistory(save_filename=filename,model_ctc=self.model_ctc,model_data=self.model_data,save_epoch=save_epoch)#这里filename要设置一样，否则保存有问题，，
            self.model_ctc.fit_generator(generator=yield_datas,
                                            steps_per_epoch=data_nums//batch_size+1 if data_nums%batch_size!=0 else data_nums//batch_size,#最后一步小于等于batch_size
                                            epochs=epochs,
                                            verbose=1,
                                            #validation_data=yield_validation,
                                            #validation_steps=50,
                                            callbacks=[hist],
                                            initial_epoch=self.epoch_all#大胆猜测是从0开始,,,错了也就是差1吧emmm
                                            )

        except StopIteration:
            raise LZ_Error("[错误QAQ]貌似生成的数据格式有点问题？？")#天知道触发吗              

        #hist.plot()
        #hist.save('lz_new.json')#这两个不能这样了先
        self.SaveModel(filename)#保存所有东西

    def SaveModel(self,filename):
        '''
        filename:保存的文件名，路径、后缀已准备好
        '''
        self.model_ctc.save(os.path.join(self.save_path,filename+'_ctc.h5'))
        self.model_data.save(os.path.join(self.save_path,filename+'_data.h5'))
        self.model_ctc.save_weights(os.path.join(self.save_path,filename+'_weights_ctc.h5'))
        self.model_data.save_weights(os.path.join(self.save_path,filename+'_weights_data.h5'))

    def TestModel(self,filename,test_size=-1,type='both'):
        '''
        type:train test dev both(test+dev)
        计算总拼音错误率
        test_size 为-1时即全部,其他即为所选类型各10
        filename:保存的文件名，路径、后缀已准备好
        '''
        deal_batch=16#emmm
        if type not in ['both','test','dev','train']:
            raise TypeError('type 不对 应为 both 或 test 或 dev 或 train')
        import tensorflow as tf
        path=os.path.join(self.save_path,filename+'_weights_data_120.h5')
        self.model_data.load_weights(path)
        speech_datas=[]
        if type=='both':
            speech_datas.append(DataSpeech(self.relpath,'dev'))
            speech_datas.append(DataSpeech(self.relpath,'test'))
        else:
            speech_datas.append(DataSpeech(self.relpath,type))

        total_distance=0
        total_pinyin=0
        for speech_data in speech_datas:
            all_test=speech_data.DataNum_Total
            X=np.zeros((deal_batch,1600,200,1),dtype=np.float64)
            input_length = np.zeros((deal_batch),dtype = np.int32)
            data_output_list=[]
            test_size=test_size if test_size>0 & test_size<=all_test else all_test
            total_count=test_size//deal_batch
            remainder=test_size%deal_batch
            from tqdm import tqdm
            print('评估计算中~')
            for i in tqdm(range(0,total_count)):
                start=i*deal_batch
                end=(i+1)*deal_batch
                for j,k in zip(range(0,deal_batch),range(start,end)):
                    data_input,data_output=speech_data.GetData(k)
                    data_output_list.append(data_output)
                    X[j,0:len(data_input)]=data_input
                    input_length[j]=(min(data_input.shape[0] // 8 + data_input.shape[0] % 8,200))
                temp=self.model_data.predict(X)
                pinyin_pre_decode=K.ctc_decode(temp,input_length,greedy=True,beam_width=100,top_paths=1,merge_repeated=False)
                #greedy改为False可能会好点但太慢了
                pinyin_pre=K.eval(pinyin_pre_decode[0][0])
                for j in range(0,deal_batch):                    
                    pinyin_pre_temp=speech_data.num2symbol(pinyin_pre[j])
                    pinyin_real=speech_data.num2symbol(data_output_list[j])
                    total_distance+=GetEditDistance_pinyin(pinyin_pre_temp,pinyin_real)
                    total_pinyin+=len(pinyin_real)
                if i%10==0:
                    K.clear_session()
                    self.CreateModel()
                    self.model_data.load_weights(path)
                data_output_list=[]
            
            if remainder!=0:
                K.clear_session()
                self.CreateModel()
                self.model_data.load_weights(path)
                X=np.zeros((remainder,1600,200,1),dtype=np.float64)
                input_length = np.zeros((remainder),dtype = np.int32)
                data_output_list=[]
                start=total_count*deal_batch
                end=test_size
                for j,k in zip(range(0,remainder),range(start,end)):
                    data_input,data_output=speech_data.GetData(k)
                    data_output_list.append(data_output)
                    X[j,0:len(data_input)]=data_input
                    input_length[j]=(min(data_input.shape[0] // 8 + data_input.shape[0] % 8,200))
                temp=self.model_data.predict(X)
                pinyin_pre_decode=K.ctc_decode(temp,input_length,greedy=True,beam_width=100,top_paths=1,merge_repeated=False)
                #greedy改为False可能会好点但太慢了
                pinyin_pre=K.eval(pinyin_pre_decode[0][0])
                for j in range(0,remainder):                
                    pinyin_pre_temp=speech_data.num2symbol(pinyin_pre[j])
                    pinyin_real=speech_data.num2symbol(data_output_list[j])
                    total_distance+=GetEditDistance_pinyin(pinyin_pre_temp,pinyin_real)
                    total_pinyin+=len(pinyin_real)
            
        print('total_WER:'+str(total_distance/total_pinyin))

    def VisualModel(self,filename):
        '''
        filename:保存的文件名，路径、后缀已准备好，如model_lz
        '''
        path='model_image'
        plot_model(self.model_ctc,to_file=os.path.join(path,'model_ctc.png'),show_shapes=True)
        plot_model(self.model_data,to_file=os.path.join(path,'model_data.png'),show_shapes=True)

class LossHistory(Callback):
    def __init__(self,save_filename,model_ctc,model_data,save_epoch=5):
        super(LossHistory,self).__init__()
        self.save_filename=save_filename
        self.model_ctc_to_save=model_ctc
        self.model_data_to_save=model_data
        self.save_epoch=save_epoch
        self.loss = {'batch':[], 'epoch':[]}
        self.acc = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
        
        self.save_path='model_save'
        system_type = plat.system()
        self.slash = ''
        if(system_type == 'Windows'):
           self.slash = '\\' # 反斜杠
        elif(system_type == 'Linux'):
           self.slash = '/' # 正斜杠
        else:
           print('[Message] Unknown System\n')
           self.slash = '/' # 正斜杠

    def on_batch_end(self, batch, logs=None):
        self.loss['batch'].append(float(logs.get('loss')))
        self.acc['batch'].append(float(logs.get('accuracy')))
        #self.val_loss['batch'].append(logs.get('val_loss'))
        #self.val_acc['batch'].append(logs.get('val_acc'))
        #好像这不对哦，fit_generator的参数介绍是下面这么写的，可测一下，，，
        #on which to evaluate
        #the loss and any model metrics at the end of each epoch.

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1)%self.save_epoch==0:#这个epoch貌似，小1？
            #这个暂时先不用 其实我一度怀疑这真的行吗？？有空试试
            self.model_ctc_to_save.save_weights(os.path.join(self.save_path,self.save_filename+'_weights_ctc_%d.h5'%(epoch+1)))
            self.model_data_to_save.save_weights(os.path.join(self.save_path,self.save_filename+'_weights_data_%d.h5'%(epoch+1)))
            self.model_ctc_to_save.save(os.path.join(self.save_path,self.save_filename+'_ctc_%d.h5'%(epoch+1)))
            self.model_data_to_save.save(os.path.join(self.save_path,self.save_filename+'_data_%d.h5'%(epoch+1)))

            path='loss_acc_save'
            with open(os.path.join(path,'batch_loss2.json'),mode='a') as file_object:
                temp=json.dumps({'loss':self.loss['batch'],'acc':self.acc['batch']})
                file_object.write(temp+'\n')
            self.loss['batch'].clear()
            self.acc['batch'].clear()

        self.loss['epoch'].append(float(logs.get('loss')))
        self.acc['epoch'].append(float(logs.get('accuracy')))
        #self.val_loss['epoch'].append(logs.get('val_loss'))
        #self.val_acc['epoch'].append(logs.get('val_acc'))
        with open(os.path.join(self.save_path,self.save_filename+'_epochs.pkl'),'wb') as f:
            pickle.dump(epoch+1,f)#保存epoch数

    def save(self,filename):
        '''保存的文件名称'''
        #这个地方还是要换行保存才行！
        path='loss_acc_save'
        with open(os.path.join(path,filename),mode='a') as file_object:
            #json.dump({'loss':self.loss,'acc':self.acc},file_object,indent=2)
            temp=json.dumps({'loss':self.loss['epoch'],'acc':self.acc['epoch']})
            file_object.write(temp+'\n')

    def plot(self,loss_type='epoch'):
        '''loss_type:batch,epoch'''
        iters = range(len(self.loss[loss_type]))
        fig=plt.figure()
        # acc
        plt.plot(iters, self.acc[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.loss[loss_type], 'g', label='train loss')
        #我没开 val...
        #if loss_type == 'epoch':
            # val_acc
        #    plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
        #    plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
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

    #print(D[len_label][len_pred])
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
    #print(leven_cost)
    return leven_cost

def ctc_loss_func(args):#这玩意要拉出来还是
    y_pred,label,input_length,label_length=args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    #y_pred = y_pred[:, 2:, :] 然后博士保留了这个垃圾hhh
    #y_pred = y_pred[:, :, :]

    return K.ctc_batch_cost(label,y_pred,input_length,label_length)

if(__name__ == '__main__'):
    #S_M=SpeechModel('dataset')
    #S_M.LoadModel('lz_new')
    #S_M.TrainModel('lz_new',epochs=200,save_epoch=1)
    #S_M.PredModel('lz_new')
    GetEditDistance_pinyin(['ni3','hao3'],['bu4','hao3','a1'])
    GetEditDistance_str('绿 是 阳春 烟 警 大 快 文章 的 底色 四月 的 林 栾 更 是 率 的 先 活 秀媚 失意 盎然',
                        '绿 是 阳春 烟 景 大块 文章 的 底色 四月 的 林 峦 更是 绿 得 鲜活 秀媚 诗意 盎然')