import argparse
import os
from warnings import warn
from time import sleep

import tensorflow as tf

from tacotron.hparams import hparams
from tacotron.infolog import log
from tacotron.synthesize import tacotron_synthesize
import wavenet_synthesis as ws
from pypinyin import pinyin, Style
from wavenet_emmm import preprocess_normalize 
import joblib
os.environ["CUDA_VISIBLE_DEVICES"]='0'

def main(text):
    #无语
    temp_list=pinyin(text,style=Style.TONE3)
    sentence=[]
    for temp in temp_list:
        sentence.append(temp[0])
    sentences=[' '+' '.join(sentence)]
    print(sentences)
    taco_checkpoint_path = os.path.join('taco_model','tacotron_model.ckpt-100000')
    wave_checkpoint_path = os.path.join('wave_model','checkpoint_latest.pth')
    taco_output_dir = 'taco_output'
    wave_output_dir = 'wave_output'

    
    _ = tacotron_synthesize(hparams, taco_checkpoint_path, os.path.join(taco_output_dir,'org'),sentences)
       
    
    for i in range(len(sentences)):
        in_dir = os.path.join(taco_output_dir,'org')
        out_dir = os.path.join(taco_output_dir,'norm')
        scaler_path = os.path.join('wavenet_emmm','meanvar.joblib')
        scaler = joblib.load(scaler_path)
        inverse = None
        num_workers = None
        from multiprocessing import cpu_count
        num_workers = cpu_count() // 2 if num_workers is None else int(num_workers)

        os.makedirs(out_dir, exist_ok=True)
        preprocess_normalize.apply_normalization_dir2dir(in_dir, out_dir, scaler, inverse, num_workers)

    for i in range(len(sentences)):
        ws.wav_syn(wave_checkpoint_path,wave_output_dir,
               os.path.join(os.path.join(taco_output_dir,'norm'),'mel-sentence_{}.npy'.format(i)))

if __name__ == '__main__':
    
   main('人生若只如初见,何事秋风悲画扇')

