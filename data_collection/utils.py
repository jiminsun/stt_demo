from datetime import datetime


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