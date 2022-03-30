import time
import argparse
import warnings
from tqdm import tqdm
import soundfile as sf
from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor

from utils import load_wav, list_wav_files, parse_index, SAMPLE_RATE
from jiwer import wer, cer
from getradio import GetRadio

warnings.filterwarnings("ignore")

MODEL_NAME = 'facebook/s2t-small-librispeech-asr'
PREFIX = 'butler traffic skyhawk seven three seven'
SUFFIX = 'butler traffic'
RECORD_TIME_IN_SECONDS = 5


class SpeechToText:
    def __init__(self, model_config=MODEL_NAME, sampling_rate=SAMPLE_RATE):
        self.model = Speech2TextForConditionalGeneration.from_pretrained(
            model_config
        )
        self.processor = Speech2TextProcessor.from_pretrained(
            model_config
        )
        self.sampling_rate = sampling_rate
        self.recorder = GetRadio()

    def map_to_array(self, batch):
        speech, _ = sf.read(batch["file"])
        batch["speech"] = speech
        return batch

    def record(self, seconds):
        self.recorder.start_recording()
        time.sleep(seconds)
        frames = self.recorder.stop_recording()
        return frames

    def load_labels(self, txt_fname):
        with open(txt_fname, 'r') as f:
            labels = [l.rstrip() for l in f.readlines()]
        labels = [' '.join([PREFIX, l, SUFFIX]) for l in labels]
        return labels

    def predict(self, data):
        inputs = self.processor(
            data,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        )
        generated_ids = self.model.generate(
            inputs=inputs["input_features"],
            attention_mask=inputs["attention_mask"]
        )
        transcription = self.processor.batch_decode(generated_ids)
        return transcription[0]


def main(args):
    # load labels
    stt_model = SpeechToText(MODEL_NAME)

    if len(list_wav_files(args.data_dir)):
        import pandas as pd

        # load model
        # get test audio
        indices = []
        predictions = []
        labels = []
        word_error_rate = []
        char_error_rate = []
        for fname in tqdm(list_wav_files(args.data_dir)):
            audio = load_wav(fname, normalize=True)
            audio_idx = parse_index(fname)
            prediction = stt_model.predict(audio)

            # Append results
            indices.append(audio_idx)
            predictions.append(prediction)

        prediction_df = pd.DataFrame({
            'index': indices,
            'prediction': predictions,
        }).sort_values(by='index').reset_index(drop=True)

        if args.label is not None:
            idx2label = stt_model.load_labels(args.label)
            for fname in tqdm(list_wav_files(args.data_dir)):
                audio_idx = parse_index(fname)
                # Append results
                indices.append(audio_idx)
                label = idx2label[audio_idx - 1]
                labels.append(label)  # audio idx starts with 0

            for (label, prediction) in zip(labels, predictions):
                word_error_rate.append(wer(label, prediction))
                char_error_rate.append(cer(label, prediction))

            prediction_df['label'] = labels
            prediction_df['wer'] = word_error_rate
            prediction_df['cer'] = char_error_rate

            assert len(indices) == len(predictions) == len(labels)


        prediction_df.to_csv(f'{args.output_fname}', sep='\t')
        print(f'Outputs saved as {args.output_fname}')

    elif args.test_file is not None:
        wav_file = load_wav(args.test_file, normalize=True)
        text = stt_model.predict(wav_file)
        print("===== Transcribed output ")
        print(text)

    else:
        data = stt_model.record(RECORD_TIME_IN_SECONDS)
        text = stt_model.predict(data)
        print("===== Transcribed output ")
        print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_file",
        default=None,
        type=str,
        help='wav file to test speech transcription'
    )
    parser.add_argument(
        "--data_dir",
        default='./test_data',
        type=str,
        help='directory containing test wav files'
    )
    parser.add_argument(
        '--output_fname',
        default='prediction.csv',
        type=str,
        help='output filename to write test predictions'
    )
    parser.add_argument(
        '--label',
        default='input.txt',
        type=str,
        help='txt file containing transcription labels'
    )

    args = parser.parse_args()

    main(args)
