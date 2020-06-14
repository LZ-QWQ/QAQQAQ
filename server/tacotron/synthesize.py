import argparse
import os
import re
import time
from time import sleep

import tensorflow as tf
from tacotron.hparams import hparams, hparams_debug_string
from tacotron.infolog import log
from tacotron.synthesizer import Synthesizer
from tqdm import tqdm


def run_eval(checkpoint_path, output_dir, hparams, sentences):
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

    #Set inputs batch wise
    sentences = [sentences[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]

    log('Starting Synthesis')
    with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
        for i, texts in enumerate(tqdm(sentences)):
            start = time.time()
            #basenames = ['sentence_{}'.format(i, j) for j in range(len(texts))]
            basenames = ['sentence_{}'.format(i)]
            mel_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir,log_dir, None)

            for elems in zip(texts, mel_filenames, speaker_ids):
                file.write('|'.join([str(x) for x in elems]) + '\n')
    log('synthesized mel spectrograms at {}'.format(eval_dir))
    return eval_dir

def tacotron_synthesize(hparams, checkpoint_path,output_dir,sentences=None):       
    log('loaded model at {}'.format(checkpoint_path))

    if hparams.tacotron_synthesis_batch_size < hparams.tacotron_num_gpus:
        raise ValueError('Defined synthesis batch size {} is smaller than minimum required {} (num_gpus)! Please verify your synthesis batch size choice.'.format(
            hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

    if hparams.tacotron_synthesis_batch_size % hparams.tacotron_num_gpus != 0:
        raise ValueError('Defined synthesis batch size {} is not a multiple of {} (num_gpus)! Please verify your synthesis batch size choice!'.format(
            hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

    return run_eval(checkpoint_path, output_dir, hparams, sentences)

