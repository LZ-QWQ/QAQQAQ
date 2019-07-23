from nl_speechmodel2 import SpeechModel

S_M=SpeechModel('dataset')
S_M.LoadModel('lz_test2')
S_M.TrainModel('lz_test2',epochs=200)
#S_M.PredModel('lz_test2')