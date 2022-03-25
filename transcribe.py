import argparse
import pandas as pd
from tqdm import tqdm
import soundfile as sf
from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor

from utils import load_wav, list_wav_files, parse_index, SAMPLE_RATE
from jiwer import wer, cer


MODEL_NAME = 'facebook/s2t-small-librispeech-asr'
PREFIX = 'butler traffic skyhawk seven three seven'
SUFFIX = 'butler traffic'


def load_pretrained(model_class, processor_class, pretrained_name):
    model = model_class.from_pretrained(pretrained_name)
    processor = processor_class.from_pretrained(pretrained_name)
    return model, processor


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


def load_labels(txt_fname):
    with open(txt_fname, 'r') as f:
        labels = [l.rstrip() for l in f.readlines()]
    labels = [' '.join([PREFIX, l, SUFFIX]) for l in labels]
    return labels


def predict(data, processor, model):
    inputs = processor(data, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    generated_ids = model.generate(
        inputs=inputs["input_features"],
        attention_mask=inputs["attention_mask"]
    )
    transcription = processor.batch_decode(generated_ids)
    return transcription[0]


def main(args):
    # load labels
    idx2label = load_labels(args.label)

    # load model
    model, processor = load_pretrained(
        model_class=Speech2TextForConditionalGeneration,
        processor_class=Speech2TextProcessor,
        pretrained_name=MODEL_NAME
    )
    # get test audio
    indices = []
    predictions = []
    labels = []
    word_error_rate = []
    char_error_rate = []
    for fname in tqdm(list_wav_files(args.data_dir)):
        audio = load_wav(fname, normalize=True)
        audio_idx = parse_index(fname)
        prediction = predict(audio, processor, model)

        # Append results
        indices.append(audio_idx)
        predictions.append(prediction)
        label = idx2label[audio_idx-1]
        labels.append(label)   # audio idx starts with 0
        word_error_rate.append(wer(label, prediction))
        char_error_rate.append(cer(label, prediction))


    assert len(indices) == len(predictions) == len(labels)

    prediction_df = pd.DataFrame({
        'index': indices,
        'prediction': predictions,
        'label': labels,
        'wer': word_error_rate,
        'cer': char_error_rate
    }).sort_values(by='index').reset_index(drop=True)
    prediction_df.to_csv(f'{args.output_fname}', sep='\t')
    print(f'Outputs saved as {args.output_fname}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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