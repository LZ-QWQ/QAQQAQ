from nl_speechmodel2_new import SpeechModel
import os
S_M=SpeechModel('dataset')
filename='lz_new'
if os.path.exists(filename+'_weights_ctc.h5') and \
    os.path.exists(filename+'_weights_ctc.h5'):
    S_M.LoadModel(filename)
S_M.TrainModel(filename,epochs=200,save_epoch=1)
#S_M.PredModel('lz_new')