import numpy as np
import tensorflow as tf
import time
import librosa
import sounddevice as sd
import matplotlib.pyplot as plt

interpreter = tf.lite.Interpreter(model_path=r"D:\python\safevision\SafeVision\last snoring detection\src\convert\quant_28_128_breath_2.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

SAMPLING_RATE = 44100
MFCC_NUM = 28
MFCC_MAX_LEN = 128
SECONDS_PER_SLICE = 1

def record_audio(duration, sampling_rate):
    #print(f"Recording {duration} seconds...")
    audio = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1, dtype='float32', device = 2)
    sd.wait()
    return audio.flatten()

def preprocess_and_predict(audio):
    spectrogram = librosa.stft(audio, n_fft=320, hop_length=32)
    spectrogram = np.abs(spectrogram)

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(spectrogram), n_mfcc=MFCC_NUM, sr=SAMPLING_RATE)

    if mfcc.shape[1] < MFCC_MAX_LEN:
        pad_width = MFCC_MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MFCC_MAX_LEN]

    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = np.expand_dims(mfcc, axis=0)

    scale, zero_point = input_details[0]['quantization']
    mfcc = ((mfcc / scale) + zero_point).astype(np.int8)
    
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], mfcc)
    interpreter.invoke()
    print(f"Processed in {time.time() - start:.4f} seconds.")
    
    scale0, zero_point0 = output_details[0]['quantization']
    prediction = (interpreter.get_tensor(output_details[0]['index']).astype('float32') - zero_point0) * scale0

    print(f"Prediction: {prediction}")
    # if prediction[0] < 0.5:
    #     pass
    #     #print('Not snoring')
    # else:
    #     print('Snoring')


print("Run")

while True:
    audio = record_audio(SECONDS_PER_SLICE, SAMPLING_RATE)

    preprocess_and_predict(audio)

