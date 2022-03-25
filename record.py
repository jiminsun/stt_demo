import os
import time
import queue
import argparse
from data_collection.utils import *
from data_collection.record_audio import listen, write_wav_file


def main(args):
    with open(args.input_file, 'r') as f:
        intents = [l.rstrip() for l in f.readlines()]

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    q = queue.Queue()
    for i in range(args.num_repeat):
        for example_idx, intent in enumerate(intents, 1):
            utterance = format_utterance(
                intent=intent,
                speaker=args.speaker,
                prefix=args.prefix,
                suffix=args.suffix
            )
            print("\n")
            print(f"===== EXAMPLE {example_idx} / {len(intents)} (Iteration {i+1})")
            read_this_label_please(utterance)
            start_recording = input(f'Press enter to start recording')
            if start_recording == '':
                time.sleep(0.2)
                print('===== start speaking')
                audio = listen(q)
                output_fname = os.path.join(args.output_dir, format_output(example_idx=example_idx))
                save_file = input(f'Save recording ([y]/n)?')
                if save_file.lower() != 'n':
                    write_wav_file(audio, output_fname)
            continue_next = input(f'Press enter to proceed to next example')
            if continue_next == '':
                pass
            else:
                return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parameters to be specified for recording
    parser.add_argument(
        '--input_file',
        default='input.txt',
        type=str,
        help='input csv file with utterances'
    )
    parser.add_argument(
        '--num_repeat',
        default=10,
        type=int,
        help='number of times to repeat recording each example'
    )
    parser.add_argument(
        '--output_dir',
        default='recording',
        type=str,
        help='directory to save the recordings'
    )

    # intent specific parameters
    parser.add_argument(
        '--speaker',
        default='skyhawk 737',
        type=str,
        help='who is speaking'
    )
    parser.add_argument(
        '--prefix',
        default='butler',
        type=str,
        choices=['butler traffic', 'butler', 'butler county'],
        help='to whom the pilot is speaking, mentioned in the *beginning*'
    )
    parser.add_argument(
        '--suffix',
        default='butler',
        type=str,
        choices=['butler traffic', 'butler', 'butler county'],
        help='to whom the pilot is speaking, mentioned at the *end*'
    )

    args = parser.parse_args()
    main(args)