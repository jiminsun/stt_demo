import argparse

import torch.cuda

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd
import sounddevice as sd
from scipy.io.wavfile import write
from pydub.playback import play
from pydub import AudioSegment

# https://github.com/pytorch/fairseq/blob/main/examples/speech_synthesis/docs/ljspeech_example.md


class TextToSpeech:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        models, config, task = load_model_ensemble_and_task_from_hf_hub(
            'facebook/fastspeech2-en-ljspeech',
            arg_overrides={
                'vocoder': 'hifigan',
                'fp16': False
            }
        )
        print(type(models))
        print(models)
        self.model = models[0]
        self.model = self.model.to(self.device)
        self.task = task
        TTSHubInterface.update_cfg_with_data_cfg(
            config,
            task.data_cfg
        )
        self.generator = task.build_generator(
            self.model,
            config
        )

    def predict(self, input_text):
        sample = TTSHubInterface.get_model_input(
            self.task,
            input_text
        )

        wav, rate = TTSHubInterface.get_prediction(
            self.task,
            self.model,
            self.generator,
            sample
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