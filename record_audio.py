import time
import queue
import struct
import pyaudio

from array import array
from collections import deque


# const values for mic streaming
SR = 16000
CHUNK = 1024
BUFF = CHUNK * 10
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# const valaues for silence detection
SILENCE_THRESHOLD = 500
SILENCE_SECONDS = 2


# Import keyboard and set accepted key

def write_wav_header(_bytes, _nchannels, _sampwidth, _framerate):
    WAVE_FORMAT_PCM = 0x0001
    initlength = len(_bytes)
    bytes_to_add = b'RIFF'

    _nframes = initlength // (_nchannels * _sampwidth)
    _datalength = _nframes * _nchannels * _sampwidth

    bytes_to_add += struct.pack(
        '<L4s4sLHHLLHH4s',
        36 + _datalength, b'WAVE', b'fmt ', 16,
        WAVE_FORMAT_PCM, _nchannels, _framerate,
        _nchannels * _framerate * _sampwidth,
        _nchannels * _sampwidth,
        _sampwidth * 8, b'data'
    )

    bytes_to_add += struct.pack('<L', _datalength)

    return bytes_to_add + _bytes


def write_wav_file(stream, filename):
    try:
        file_path = f'{filename}.wav'
        wav_bytes = write_wav_header(
            stream, CHANNELS, pyaudio.PyAudio().get_sample_size(pyaudio.paInt16), SR
        )
        with open(file_path, mode='bw') as f:
            f.write(wav_bytes)
        print(f'===== wav file saved: {file_path}')
    except Exception as e:
        raise Exception(f'wav write error: {e}')


# define listen function for THREding
def listen(q):
    # open stream
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    # FIXME: release initial noisy data (1sec)
    # for _ in range(0, int(RATE / CHUNK)):
    #    data = stream.read(CHUNK, exception_on_overflow=False)

    is_started = True
    vol_que = deque(maxlen=SILENCE_SECONDS)

    while True:
        try:
            # define temporary variable to store sum of volume for 1 second
            vol_sum = 0

            # read data for 1 second in chunk
            for _ in range(0, int(RATE / CHUNK)):
                data = stream.read(CHUNK, exception_on_overflow=False)

                # get max volume of chunked data and update sum of volume
                vol = max(array('h', data))
                vol_sum += vol

                # if status is listening, check the volume value
                # if not is_started:
                #     if vol >= SILENCE_THRESHOLD:
                #         print('===== start of speech detected')
                #         is_started = True

                # if status is speech started, write data
                if is_started:
                    q.put(data)

            # if status is speech started, update volume queue and check silence
            if is_started:
                vol_que.append(vol_sum / (RATE / CHUNK) < SILENCE_THRESHOLD)
                if len(vol_que) == SILENCE_SECONDS and all(vol_que):
                    print('===== end of speech detected')
                    break

        except queue.Full:
            pass

    # close stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    start = b''
    while True:
        try:
            chunk = q.get(block=False)
            if chunk is None:
                return
            start += chunk
        except queue.Empty:
            break
    return start


if __name__ == "__main__":
    q = queue.Queue()
    start_recording = input('start recording? (Press enter to start)')
    if start_recording == "":
        time.sleep(0.2)
        print('===== start speaking')
        audio = listen(q)
        write_wav_file(audio, 'test')
