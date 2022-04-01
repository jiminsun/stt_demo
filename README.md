# stt_demo


## Running Speech-to-text

```
pip install -r requirements.txt
```


1. Running with a pre-recorded wav file
```
python stt.py --test_file $WAV_FILE
```

2. Recording and transcribing on the fly


This will record your speech for five seconds by default.
```
python stt.py
```

## Running Text-to-speech
```
bash install.sh
```

1. Run
```
python tts.py --input_text $TEXT_STRING
```
