import numpy as np 
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
import matplotlib.collections as collections
import matplotlib.patches as mpatches

from tensorflow.keras.models import load_model

MFCC_NUM =  14
SAMPLING_RATE = 44100 
MFCC_MAX_LEN = 128 

def wav2mfcc(wave, max_len=MFCC_MAX_LEN):
    mfcc = librosa.feature.mfcc(y=wave, n_mfcc=MFCC_NUM, sr=SAMPLING_RATE)
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc

def split_audio_into_segments(audio_file, segment_duration=1.0):

    audio, sr = librosa.load(audio_file, sr=None)

    segment_length = int(segment_duration * sr)

    segments = []
    for start in range(0, len(audio), segment_length):
        end = start + segment_length
        segment = audio[start:end]
        segments.append(segment)
        
    return segments, sr

def predict_for_segments(segments, model):
    predictions = []
    for i, segment in enumerate(segments):
        mfcc = wav2mfcc(segment)

        input_data = np.expand_dims(mfcc, axis=0)
 
        prediction = model.predict(input_data, verbose=0)
        prediction = 1 if prediction > 0.9 else 0
        predictions.append(prediction)
    
    return predictions

def predict_audio_file(audio_file, model):
    segments, sr = split_audio_into_segments(audio_file, segment_duration=1.0)

    predictions = predict_for_segments(segments, model)
    
    return predictions


model_path = r'D:\python\safevision\SafeVision\breath detection\models\h5\model_for_BIG_DATASET.h5'
model = load_model(model_path)

FILE_PATH = r'D:\python\safevision\SafeVision\3.wav'
pred = predict_audio_file(FILE_PATH, model)

print(pred)