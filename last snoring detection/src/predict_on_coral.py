import numpy as np
import tensorflow as tf
import time
import librosa
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import set_input
from pycoral.adapters.classify import get_classes

# Загрузка вашей квантованной модели TFLite для Coral
interpreter = make_interpreter("full_int8.tflite")
interpreter.allocate_tensors()

# Получение информации о входных и выходных тензорах
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Параметры для обработки аудио
SAMPLING_RATE = 44100
MFCC_NUM = 14
MFCC_MAX_LEN = 1000

# Функция для обработки и предсказания
def preprocess_and_predict(AUDIO_PATH):
    # Загрузка и обработка аудио
    audio, sr = librosa.load(AUDIO_PATH, sr=SAMPLING_RATE)
    mfcc = librosa.feature.mfcc(y=audio, n_mfcc=MFCC_NUM, sr=SAMPLING_RATE)
    
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
    start = time.time()
    # Устанавливаем входные данные и запускаем модель на Coral
    set_input(interpreter, mfcc)
    interpreter.invoke()
    print(time.time() - start)
    # Получаем результат и масштабируем предсказание
    scale0, zero_point0 = output_details[0]['quantization']
    age = (interpreter.tensor(output_details[0]['index'])().astype('float32') - zero_point0) * scale0

    # Вывод предсказания
    print("Prediction:", age)
    if age[0] < 0.5:
        print('не храпит')
    else:
        print('храпит')


AUDIO_PATH = '0_6.wav'
preprocess_and_predict(AUDIO_PATH)
