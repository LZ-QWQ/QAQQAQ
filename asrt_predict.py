from keras.models import load_model
import numpy as np
import platform as plat
import os
from readdata import DataSpeech

def predict(modelname):
        
        system_type = plat.system()
        slash = ''
        if(system_type == 'Windows'):
           slash = '\\' # 反斜杠
        elif(system_type == 'Linux'):
           slash = '/' # 正斜杠
        else:
           print('[Message] Unknown System\n')
           slash = '/' # 正斜杠
        
        save_path='model_save'+slash#保存模型的路径

        model_data=load_model(modelname)

        data=DataSpeech('dataset','train')
        data_input,data_label=data.GetData(11)
        X=np.zeros((1,1600,200,1),dtype=np.float64)
        X[0,0:len(data_input)]=data_input;
        y_pre=model_data.predict(X)
        print(y_pre.shape)
        y_pre.reshape((200,1424))
        for i in range(200):
            symbol=data.num2symbol(y_pre[i,:])
if(__name__ == '__main__'):
    predict('lz_test1_data.h5')