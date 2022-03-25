from scipy.io import wavfile
import numpy as np
import glob
import os


SAMPLE_RATE = 16000


def load_wav(fname, normalize=False):
    sr, data = wavfile.read(fname)
    assert sr == SAMPLE_RATE
    if normalize:
        data = normalize_audio(data)
    return data


def normalize_audio(data):
    return data.astype(np.float32, order='C') / 32768.0


def list_wav_files(fpath):
    return glob.glob(os.path.join(fpath, '*.wav'))


def parse_index(wav_fpath):
    fname = wav_fpath.split('/')[-1]
    return int(fname.split('_')[1])