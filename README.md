# AITF Speech-to-text

## 1. Installation
```
pip install -r requirements.txt
```

## 2. Test run
```
python record_audio.py
```

End of the speech will be detected automatically if the audio input is empty for two seconds (You may adjust this threshold in `record_audio.py` by changing the variable `SILENCE_SECONDS`). The recorded audio file of this test run will be saved as `test.wav`. Please check the output to see if the audio file is recorded correctly, before proceeding to the actual samples!

## 3. Recording
```
python main.py
```

The examples to read out aloud will be displayed on the command line. There are a total of 12 intent samples to be recorded, and the default setting is to repeat all sample 10 times (adjust with `--num_repeat` if needed). The speaker of the aircraft (e.g., skyhawk 737), the airport name (e.g., butler / butler county) can also be changed using the corresponding flags (`--speaker`, `--prefix`, `--suffix`. 

All outputs will be saved to the `recording` directory.
