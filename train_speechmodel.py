from nl_speechmodel2_new import SpeechModel
import os
import platform as plat

system_type = plat.system()
slash = ''
if(system_type == 'Windows'):
    slash = '\\' # 反斜杠
elif(system_type == 'Linux'):
    slash = '/' # 正斜杠
else:
    print('[Message] Unknown System\n')
    slash = '/' # 正斜杠

S_M=SpeechModel('dataset')
filename='lz_new'
if os.path.exists('model_save'+slash+filename+'_weights_ctc.h5') and \
    os.path.exists('model_save'+slash+filename+'_weights_ctc.h5'):
    S_M.LoadModel(filename)
S_M.TrainModel(filename,epochs=200,save_epoch=1)
#S_M.PredModel('lz_new')