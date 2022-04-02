sudo apt-get update
sudo apt-get install -y libportaudio2 ffmpeg


STT_ROOT=$PWD
pip install -r requirements.txt

if [ -f "$STT_ROOT/fairseq" ]; then
  rm -rf $STT_ROOT/fairseq
fi
  
git clone https://github.com/pytorch/fairseq.git
cd fairseq
pip install --editable ./


# Overwrite speech_generator.py with the edited version
cd $STT_ROOT
pip install sounddevice
pip install g2p-en

cp speech_generator.py fairseq/fairseq/
