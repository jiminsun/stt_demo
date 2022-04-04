import argparse
import ast
import os
import torch.cuda
import yaml
import pickle
from argparse import Namespace

import tts
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub, load_model_ensemble_and_task
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from fairseq.models.text_to_speech.vocoder import GriffinLimVocoder, HiFiGANVocoder
from fairseq.models.text_to_speech.fastspeech2 import FastSpeech2Model
from fairseq.speech_generator import AutoRegressiveSpeechGenerator
from scipy.io.wavfile import write
from pydub.playback import play
from pydub import AudioSegment
from pathlib import Path
from fairseq.data.audio.data_cfg import S2TDataConfig

# https://github.com/pytorch/fairseq/blob/main/examples/speech_synthesis/docs/ljspeech_example.md

CKPT_DIR = './fastspeech2_model'


class TextToSpeech:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tts_interface = FastSpeech2Model.from_pretrained(
            model_name_or_path=CKPT_DIR,
            checkpoint_file="pytorch_model.pt",
            data_name_or_path=".",
            config_yaml="config.yaml",
            vocoder="hifigan",
            fp16=False,
        )

    def predict(self, input_text):
        wav, rate = self.tts_interface.predict(
            input_text
        )
        wav = wav.cpu().numpy()
        audio_segment = AudioSegment(
            wav.tobytes(),
            frame_rate=rate,
            sample_width=wav.dtype.itemsize,
            channels=1
        )
        print("Playing the audio ...")
        play(audio_segment)
        return wav, rate


def main(args):
    tts_model = TextToSpeech()
    wav, rate = tts_model.predict(args.input_text)
    write('test_tts.wav', rate=rate, data=wav)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_text",
        default="Butler skyhawk seven three seven turning left downwind butler traffic.",
        type=str,
        help='wav file to test speech transcription'
    )

    args = parser.parse_args()
    main(args)