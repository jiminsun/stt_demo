from datetime import datetime
from scipy.io import wavfile
import numpy as np
import glob
import os


SAMPLE_RATE = 16000


def format_utterance(intent: str,
                     speaker: str,
                     prefix: str,
                     suffix: str):
    return ' '.join([prefix, speaker, intent, suffix])


def read_this_label_please(sentence: str):
    print("=" * 60)
    print(sentence)
    print("=" * 60)


def format_output(example_idx):
    now = datetime.now()
    current_time = now.strftime("%m%d%y_%H%M%S")
    return f'stt_{example_idx}_{current_time}'


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