from model_vgg_ctc import SpeechModel
import os
import platform as plat

#在此可补一个显存分配的设置
os.environ["CUDA_VISIBLE_DEVICES"]='0'
temp=os.environ["CUDA_VISIBLE_DEVICES"]
gpu_nums=len(temp.split(','))

S_M=SpeechModel('dataset')
filename='QAQ'

#保存模型图（说实话丑死了）
if not (os.path.exists(os.path.join('model_image','_ctc.png')) or \
    os.path.exists(os.path.join('model_image',filename+'_data.png'))):
    S_M.VisualModel(filename)

#S_M.PredModel(filename)
S_M.TestModel(filename,1000,type='train')

