import numpy as np
import sounddevice as sd

SR = 16000
CHUNK = 1024
BUFF = CHUNK * 10
CHANNELS = 1
RATE = 16000


class GetRadio:

    def __init__(self):
        self.stream = sd.InputStream(
            samplerate=SR,
            channels=CHANNELS,
            dtype='float32',
            callback=self.callback
        )
        self.frames = []
        self.is_record = False

    def start_recording(self):
        if not self.is_record:
            print("===== Start recording")
            self.frames = []
            self.stream.start()
            self.is_record = True

    def stop_recording(self):
        if self.is_record:
            print("===== End recording")
            self.stream.stop()
            # filename = "test.wav"
            self.is_record = False
            f = np.concatenate(self.frames, axis=0)
            f = f.squeeze(-1)
            return f

    def callback(self, in_data, frame_count, time_info, status):
        self.frames.append(in_data.copy())