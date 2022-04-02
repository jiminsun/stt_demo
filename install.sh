sudo apt-get install -y libportaudio2 ffmpeg


ROOT=$PWD
pip install -r requirements.txt

git clone https://github.com/pytorch/fairseq.git
cd fairseq
pip install --editable ./


# Overwrite speech_generator.py with the edited version
cd $ROOT
pip install sounddevice
pip install g2p-en

cp speech_generator.py fairseq/fairseq/
