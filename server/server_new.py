from flask import Flask,Response,request,send_file
from flask_restful import reqparse, abort, Api, Resource
import argparse
import os
import re
import time
from time import sleep

import tensorflow as tf
from tacotron.hparams_emmm import hparams, hparams_debug_string
from tacotron.infolog import log
from tacotron.synthesizer import Synthesizer
from tqdm import tqdm
from pypinyin import pinyin, Style
checkpoint_path = os.path.join('taco_model2','tacotron_model.ckpt-100000')
output_dir = os.path.join('taco_output','org')
eval_dir = output_dir
log_dir = os.path.join(output_dir, 'logs-eval')
    #Create output path if it doesn't exist
os.makedirs(eval_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)
log(hparams_debug_string())
synth = Synthesizer()
synth.load(checkpoint_path, hparams)


from asr_model.model_vgg_ctc import SpeechModel
import os
import platform as plat

#下面这个分配真的是无效的，我也不知道该咋使用才行，脑壳疼呢，
#而且两个模型同时部署总是会有点麻烦，打扰了，想放在cpu上也放不到，谁有能力谁搞吧，
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   
session = tf.Session(config=config)
KTF.set_session(session)

S_M=SpeechModel('dataset')
graph = tf.get_default_graph()
import language_model
LM=language_model.Language_Model('lan_model')


app=Flask('QAQ')
api = Api(app)


@app.route('/synthesis',methods = ["POST"])
def synthesis():
    def del_files(path_file):
        ls = os.listdir(path_file)
        for i in ls:
            f_path = os.path.join(path_file, i)
            os.remove(f_path)
    path_file='taco_output/org/logs-eval/wavs/'#这玩意就固定别改啦
    del_files(path_file)
    data=request.form['text']
    print(data)
    sentences=[]
    #以后下做标点切分后转拼音后转音频再合并音频
    import re
    pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
    result_list=re.split(pattern,data)
    print(result_list)
    for result in result_list:
        if result=='':continue
        temp_list=pinyin(result,style=Style.TONE3)
        sentence=[]
        for temp in temp_list:
            sentence.append(temp[0])
        sentences.append(' '.join(sentence))
    print(sentences)    
    #大于1的batch_size时候会有bug
    sentences = [sentences[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]    
    log('Starting Synthesis')       
    print(sentences)
    with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
        for i, texts in enumerate(tqdm(sentences)):
            start = time.time()
            basenames = ['sentence_{}'.format(i)]
            #basenames = ['batch_{}_sentence_{}'.format(i, j) for j in range(len(texts))]
            mel_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir,log_dir, None)

            for elems in zip(texts, mel_filenames, speaker_ids):
                file.write('|'.join([str(x) for x in elems]) + '\n')
    

    path_read_folder='taco_output/org/logs-eval/wavs/'
    path_write_wav_file='taco_output\\org\\logs-eval\\wavs\\all_temp.wav'
    def merge_files(path_read_folder, path_write_wav_file):
        import scipy.io.wavfile as wav_
        import glob
        import numpy as np
        files = os.listdir(path_read_folder)
        merged_signal = []
        for filename in glob.glob(os.path.join(path_read_folder, '*linear.wav')):
            # print(filename)
            sr, signal = wav_.read(filename)
            merged_signal.append(signal)
        merged_signal=np.hstack(merged_signal)
        merged_signal = np.asarray(merged_signal, dtype=np.int16)
        wav_.write(path_write_wav_file, sr, merged_signal)
    merge_files(path_read_folder, path_write_wav_file)

    return send_file('taco_output\\org\\logs-eval\\wavs\\all_temp.wav')

@app.route('/synthesis2',methods = ["POST"])
def synthesis2():#这个是处理输入都是拼音的情况
    sentences=request.form['text']
    sentences=[sentences]#emmm
    print(sentences)    
    #大于1的batch_size时候会有bug
    sentences = [sentences[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]    
    log('Starting Synthesis')       
    print(sentences)
    with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
        for i, texts in enumerate(tqdm(sentences)):
            start = time.time()
            basenames = ['sentence_{}'.format(i)]
            #basenames = ['batch_{}_sentence_{}'.format(i, j) for j in range(len(texts))]
            mel_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir,log_dir, None)

            for elems in zip(texts, mel_filenames, speaker_ids):
                file.write('|'.join([str(x) for x in elems]) + '\n')
    

    path_read_folder='taco_output/org/logs-eval/wavs/'
    path_write_wav_file='taco_output\\org\\logs-eval\\wavs\\all_temp.wav'
    def merge_files(path_read_folder, path_write_wav_file):
        import scipy.io.wavfile as wav_
        import glob
        import numpy as np
        files = os.listdir(path_read_folder)
        merged_signal = []
        for filename in glob.glob(os.path.join(path_read_folder, '*linear.wav')):
            # print(filename)
            sr, signal = wav_.read(filename)
            merged_signal.append(signal)
        merged_signal=np.hstack(merged_signal)
        merged_signal = np.asarray(merged_signal, dtype=np.int16)
        wav_.write(path_write_wav_file, sr, merged_signal)
    merge_files(path_read_folder, path_write_wav_file)

    return send_file('taco_output\\org\\logs-eval\\wavs\\all_temp.wav')
    
@app.route('/asr',methods = ["POST"])
def asr():
    wav=request.files['file'].read()
    with open('asr_test_file\\temp.wav', 'wb') as file: #保存到本地的文件名
        file.write(wav)
    global graph
    with graph.as_default():
        pinyin=S_M.predict('asr_test_file\\temp.wav')
    if pinyin!=False:
        chinese=LM.decode(pinyin)
        return ' '.join(pinyin)+'\n'+chinese
    return 'error'

if __name__ == '__main__':

    app.run(debug=False,host="0.0.0.0")