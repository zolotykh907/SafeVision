import tensorflow as tf
import numpy as np
from scipy import signal
import sounddevice as sd
import matplotlib.pyplot as plt
import time

# Путь к модели
MODEL_PATH = "D:\python\safevision\SafeVision\snoring detection by Igor'\source\models\small_model.tflite"

# Настройки аудиозаписи
SAMPLE_RATE = 16000  # Частота дискретизации 16 кГц
DURATION = 1  # Длительность записи 1 секунда

# Загрузка модели и настройка для использования на CPU
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
scale, z_point = input_details[0]['quantization']
target_size = (input_shape[1], input_shape[2])


def preprocess_audio(audio_data, sample_rate, target_size):
    if sample_rate != 16000:
        num_samples = int(len(audio_data) * 16000 / sample_rate)
        audio_data = signal.resample(audio_data, num_samples)
    
    audio_data = audio_data[:16000]
    if len(audio_data) < 16000:
        audio_data = np.pad(audio_data, (0, 16000 - len(audio_data)), mode='constant')
    
    audio_data = tf.convert_to_tensor(audio_data, dtype=tf.float32)
    spectrogram = tf.signal.stft(audio_data, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)  # [time, freq, 1]
    return spectrogram

def predict(audio_data):
    spectrogram = preprocess_audio(audio_data, SAMPLE_RATE, target_size)
    spectrogram = spectrogram.numpy()
    spectrogram = ((spectrogram / scale) + z_point).astype(np.int8)

    # Предсказание
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], [spectrogram])
    interpreter.invoke()
    scale0, z_point0 = output_details[0]['quantization']
    output_data = (interpreter.get_tensor(output_details[0]['index']).astype('float32') - z_point0) * scale0
    prediction = output_data

    print("Predict:", prediction)
    print("Time:", time.time() - start)

if __name__ == "__main__":
    while True:
        print("Recording...")
        audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio_data = np.squeeze(audio_data)
        predict(audio_data)
        #time.sleep(1.5)
