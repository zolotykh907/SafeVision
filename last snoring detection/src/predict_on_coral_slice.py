import numpy as np
import tensorflow as tf
import time
import librosa
import matplotlib
from matplotlib import pyplot as plt
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import set_input
from pycoral.adapters.classify import get_classes

# Загрузка вашей квантованной модели TFLite для Coral
interpreter = make_interpreter("quant_model_14_128.tflite")
interpreter.allocate_tensors()

# Получение информации о входных и выходных тензорах
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Параметры для обработки аудио
SAMPLING_RATE = 44100
MFCC_NUM = 14
MFCC_MAX_LEN = 128
SECONDS_PER_SLICE = 1  # Размер среза в секундах

# Функция для обработки и предсказания
def preprocess_and_predict(AUDIO_PATH):
    # Загрузка и обработка аудио
    audio, sr = librosa.load(AUDIO_PATH, sr=SAMPLING_RATE)
    # Разделяем аудио на отрезки длиной 1 секунда
    audio_length = len(audio)
    num_samples_per_slice = SAMPLING_RATE * SECONDS_PER_SLICE
    num_slices = audio_length // num_samples_per_slice

    for slice_idx in range(num_slices):
        start_sample = slice_idx * num_samples_per_slice
        end_sample = (slice_idx + 1) * num_samples_per_slice
        audio_slice = audio[start_sample:end_sample]

        # Применяем STFT к срезу
        spectrogram = librosa.stft(audio_slice, n_fft=320, hop_length=32)
        spectrogram = np.abs(spectrogram)
        
        # Преобразуем спектрограмму в MFCC
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(spectrogram), n_mfcc=MFCC_NUM, sr=SAMPLING_RATE)
        
        # Приводим MFCC к нужной длине
        if mfcc.shape[1] < MFCC_MAX_LEN:
            pad_width = MFCC_MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MFCC_MAX_LEN]
        
        # Добавляем размерность канала и батча
        mfcc = np.expand_dims(mfcc, axis=-1)
        mfcc = np.expand_dims(mfcc, axis=0)

        # Преобразуем данные в uint8 для квантованной модели
        scale, zero_point = input_details[0]['quantization']
        mfcc = ((mfcc / scale) + zero_point).astype(np.int8)
        
        # Прогоняем данные через модель
        start = time.time()
        set_input(interpreter, mfcc)
        interpreter.invoke()
        print(f"Slice {slice_idx+1}/{num_slices} processed in {time.time() - start:.4f} seconds.")
        
        # Получаем результат и масштабируем предсказание
        scale0, zero_point0 = output_details[0]['quantization']
        prediction = (interpreter.tensor(output_details[0]['index'])().astype('float32') - zero_point0) * scale0

        # Вывод предсказания для каждого среза
        print(f"Prediction for slice {slice_idx+1}: {prediction}")
        if prediction[0] < 0.5:
            print('Не храпит')
        else:
            print('Храпит')



# Путь к вашему аудиофайлу
AUDIO_PATH = 'govno2.wav'
preprocess_and_predict(AUDIO_PATH)


