# GENEA_2022
This repo contains FineMotions's solution to the GENEA 2022 Challange

## Results
Our submission as well as some renders of our experiments could be 
found [here](https://drive.google.com/drive/folders/1fMnBO2Z1iTqqfWy_gR5VFwfGHJ-ZjkQY?usp=sharing)

## Structure
The whole repo contains code of the models along with various scripts:
- `process_motion.py` - extracts motion features from bvh-data. `pymo` required.
- `process_audio.py` - extracts features from audio
- `process_text.py` - generetes one-hot encoddings for symbols of text transcripts
- `normalize.py` - normalize motion features and store mean and max poses to npz file
- `align_data.py` - aligns motion, audio and text features to create dataset for models
- `train.py` - train one of the model by it's name: `wav2gest`, `recell`, `recellseq`, `feedforward`, `seq2seq`, `lstm`
- `infer.py` - inference model, smooth, denormalize results and generate bvh

module `src` contains code of the models and some `utils`:
- `base` contains base `DataModule` to operate with various data
- `feedforward` - simple model generates motion by frame from window of features. Based on Kucherenko et al. 2019 
- `lstm` - predicts sequence of poses from aligned sequence of features via simple rnn
- `recell` - auto-regressive ReCell model, contains two systems and datasets for one-frame generation 
and for short sequences which allows teacher-forcing and zeroing techniques
- `seq2seq` - sequence-to-sequence model from https://github.com/FineMotion/GENEA_2020
- `seqae` - unfinished autoencoder for sequences, has not been used
- `wav2gest` - modification of `seq2seq` allowing different lengths for input and output sequences 
- `auto-encoder` - windowed auto-encoder