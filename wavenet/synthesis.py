# coding: utf-8
"""
Synthesis waveform from trained WaveNet.

usage: synthesis.py [options] <checkpoint> <dst_dir>

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    --preset=<json>                   Path of preset parameters (json).
    --length=<T>                      Steps to generate [default: 32000].
    --initial-value=<n>               Initial value for the WaveNet decoder.
    --conditional=<p>                 Conditional features path.
    --file-name-suffix=<s>            File name suffix [default: ].
    --speaker-id=<id>                 Speaker ID (for multi-speaker model).
    --output-html                     Output html for blog post.
    --force-cpu                       Force CPU synthesis mode.
    -h, --help               Show help message.
"""
from docopt import docopt

import sys
import os
from os.path import dirname, join, basename, splitext
import torch
import numpy as np
from nnmnkwii import preprocessing as P
from tqdm import tqdm
import librosa

from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw

import audio
from hparams import hparams

from train import to_categorical

from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from torch.utils import data as data_utils
from evaluate import dummy_collate, to_int16
from torch.nn import functional as F
from scipy.io import wavfile

torch.set_num_threads(4)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def batch_wavegen(model, c=None, g=None, fast=True, tqdm=tqdm):
    from train import sanity_check
    sanity_check(model, c, g)
    assert c is not None
    B = c.shape[0]
    model.eval()
    if fast:
        model.make_generation_fast_()

    # Transform data to GPU
    g = None if g is None else g.to(device)
    c = None if c is None else c.to(device)

    if hparams.upsample_conditional_features:
        length = (c.shape[-1] - hparams.cin_pad * 2) * audio.get_hop_size()
    else:
        # already dupulicated
        length = c.shape[-1]

    with torch.no_grad():
        y_hat = model.incremental_forward(
            c=c, g=g, T=length, tqdm=tqdm, softmax=True, quantize=True,
            log_scale_min=hparams.log_scale_min)

    if is_mulaw_quantize(hparams.input_type):
        # needs to be float since mulaw_inv returns in range of [-1, 1]
        y_hat = y_hat.max(1)[1].view(B, -1).float().cpu().data.numpy()
        for i in range(B):
            y_hat[i] = P.inv_mulaw_quantize(y_hat[i], hparams.quantize_channels - 1)
    elif is_mulaw(hparams.input_type):
        y_hat = y_hat.view(B, -1).cpu().data.numpy()
        for i in range(B):
            y_hat[i] = P.inv_mulaw(y_hat[i], hparams.quantize_channels - 1)
    else:
        y_hat = y_hat.view(B, -1).cpu().data.numpy()

    if hparams.postprocess is not None and hparams.postprocess not in ["", "none"]:
        for i in range(B):
            y_hat[i] = getattr(audio, hparams.postprocess)(y_hat[i])

    if hparams.global_gain_scale > 0:
        for i in range(B):
            y_hat[i] /= hparams.global_gain_scale

    return y_hat


def _to_numpy(x):
    # this is ugly
    if x is None:
        return None
    if isinstance(x, np.ndarray) or np.isscalar(x):
        return x
    # remove batch axis
    if x.dim() == 3:
        x = x.squeeze(0)
    return x.numpy()


def wavegen(model, length=None, c=None, g=None, initial_value=None,
            fast=False, tqdm=tqdm):
    """Generate waveform samples by WaveNet.

    Args:
        model (nn.Module) : WaveNet decoder
        length (int): Time steps to generate. If conditinlal features are given,
          then this is determined by the feature size.
        c (numpy.ndarray): Conditional features, of shape T x C
        g (scaler): Speaker ID
        initial_value (int) : initial_value for the WaveNet decoder.
        fast (Bool): Whether to remove weight normalization or not.
        tqdm (lambda): tqdm

    Returns:
        numpy.ndarray : Generated waveform samples
    """
    from train import sanity_check
    sanity_check(model, c, g)

    c = _to_numpy(c)
    g = _to_numpy(g)

    model.eval()
    if fast:
        model.make_generation_fast_()

    if c is None:
        assert length is not None
    else:
        # (Tc, D)
        if c.ndim != 2:
            raise RuntimeError(
                "Expected 2-dim shape (T, {}) for the conditional feature, but {} was actually given.".format(hparams.cin_channels, c.shape))
            assert c.ndim == 2
        Tc = c.shape[0]
        upsample_factor = audio.get_hop_size()
        # Overwrite length according to feature size
        length = Tc * upsample_factor
        # (Tc, D) -> (Tc', D)
        # Repeat features before feeding it to the network
        if not hparams.upsample_conditional_features:
            c = np.repeat(c, upsample_factor, axis=0)

        # B x C x T
        c = torch.FloatTensor(c.T).unsqueeze(0)

    if initial_value is None:
        if is_mulaw_quantize(hparams.input_type):
            initial_value = P.mulaw_quantize(0, hparams.quantize_channels - 1)
        else:
            initial_value = 0.0

    if is_mulaw_quantize(hparams.input_type):
        assert initial_value >= 0 and initial_value < hparams.quantize_channels
        initial_input = np_utils.to_categorical(
            initial_value, num_classes=hparams.quantize_channels).astype(np.float32)
        initial_input = torch.from_numpy(initial_input).view(
            1, 1, hparams.quantize_channels)
    else:
        initial_input = torch.zeros(1, 1, 1).fill_(initial_value)

    g = None if g is None else torch.LongTensor([g])

    # Transform data to GPU
    initial_input = initial_input.to(device)
    g = None if g is None else g.to(device)
    c = None if c is None else c.to(device)

    with torch.no_grad():
        y_hat = model.incremental_forward(
            initial_input, c=c, g=g, T=length, tqdm=tqdm, softmax=True, quantize=True,
            log_scale_min=hparams.log_scale_min)

    if is_mulaw_quantize(hparams.input_type):
        y_hat = y_hat.max(1)[1].view(-1).long().cpu().data.numpy()
        y_hat = P.inv_mulaw_quantize(y_hat, hparams.quantize_channels)
    elif is_mulaw(hparams.input_type):
        y_hat = P.inv_mulaw(y_hat.view(-1).cpu().data.numpy(), hparams.quantize_channels)
    else:
        y_hat = y_hat.view(-1).cpu().data.numpy()

    if hparams.postprocess is not None and hparams.postprocess not in ["", "none"]:
        y_hat = getattr(audio, hparams.postprocess)(y_hat)

    if hparams.global_gain_scale > 0:
        y_hat /= hparams.global_gain_scale

    return y_hat

class SingleFileDataSource(FileDataSource):
    def __init__(self, file_path):
        self.file_path = file_path

    def collect_files(self):
        paths = []
        paths.append(self.file_path)

    def collect_features(self, path):
        return np.load(self.file_path)

if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_path = args["<checkpoint>"]
    dst_dir = args["<dst_dir>"]

    length = int(args["--length"])
    initial_value = args["--initial-value"]
    initial_value = None if initial_value is None else float(initial_value)
    conditional_path = args["--conditional"]

    file_name_suffix = args["--file-name-suffix"]
    output_html = args["--output-html"]
    speaker_id = args["--speaker-id"]
    speaker_id = None if speaker_id is None else int(speaker_id)
    preset = args["--preset"]

    # Force CPU synthesis mode if required
    if args["--force-cpu"]:
        use_cuda = False
        device = torch.device("cpu")

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "wavenet_vocoder"

    # Load conditional features
    if conditional_path is not None:
        c = np.load(conditional_path)
        if c.shape[1] != hparams.num_mels:
            c = np.swapaxes(c, 0, 1)
    else:
        c = None

    from train import build_model

    # Model
    model = build_model().to(device)

    # Load checkpoint
    print("Load checkpoint from {}".format(checkpoint_path))
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    checkpoint_name = splitext(basename(checkpoint_path))[0]

    os.makedirs(dst_dir, exist_ok=True)
    if not file_name_suffix:
        file_name_suffix = splitext(basename(conditional_path))[0]
    dst_wav_path = join(dst_dir, "{}.wav".format(file_name_suffix))

    # Prepare mel spectrogram condition
    C = FileSourceDataset(SingleFileDataSource(conditional_path))
    data_loader = data_utils.DataLoader(C, batch_size=hparams.batch_size, drop_last=False,
        num_workers=hparams.num_workers, sampler=None, shuffle=False,
        collate_fn=dummy_collate, pin_memory=hparams.pin_memory)

    cin_pad = hparams.cin_pad

    for idx, (x, y, c, g, input_lengths) in enumerate(data_loader):
        if cin_pad > 0:
            c = F.pad(c, pad=(cin_pad, cin_pad), mode="replicate")

        # B x 1 x T
        if x[0] is not None:
            B, _, T = x.shape
        else:
            B, _, Tn = c.shape
            T = Tn * audio.get_hop_size()

        # DO generate
        y_hats = batch_wavegen(model, c=c, g=g, fast=True, tqdm=tqdm)

        for i, (ref, gen, length) in enumerate(zip(x, y_hats, input_lengths)):
            gen = gen[:length]
            gen = np.clip(gen, -1.0, 1.0)

            # save
            wavfile.write(dst_wav_path, hparams.sample_rate, to_int16(gen))

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)


