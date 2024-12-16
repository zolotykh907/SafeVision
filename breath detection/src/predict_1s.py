import numpy as np 
import matplotlib.pyplot as plt
import os
import librosa

MFCC_NUM =  14
SAMPLING_RATE = 7000 
MFCC_MAX_LEN = 128 

def wav2mfcc(wave, max_len=MFCC_MAX_LEN):
    mfcc = librosa.feature.mfcc(y=wave, n_mfcc=MFCC_NUM, sr=SAMPLING_RATE)
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc

def preprocess_audio(wave):    
    if len(wave) == 0:
        raise ValueError("Аудиофайл пуст.")

    mfcc = wav2mfcc(wave)

    return np.expand_dims(mfcc, axis=0)

def predict_from_file(wave, model):
    input_data = preprocess_audio(wave)
    
    prediction = model.predict(input_data, verbose=0)

    #print(prediction)
    return prediction
