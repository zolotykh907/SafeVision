import numpy as np
import librosa
import os
import tensorflow as tf

# Загрузка вашей обученной модели
MODEL_PATH = '/home/egor/safeVision/code/model/28x128_breath_3.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# Параметры для обработки аудио
SAMPLING_RATE = 44100
MFCC_NUM = 28
MFCC_MAX_LEN = 128

# Функция для загрузки и предобработки аудио
def load_and_preprocess_audio(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLING_RATE)
    mfcc = librosa.feature.mfcc(y=audio, n_mfcc=MFCC_NUM, sr=SAMPLING_RATE)
    
    # Приводим к нужной длине
    if mfcc.shape[1] < MFCC_MAX_LEN:
        pad_width = MFCC_MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MFCC_MAX_LEN]
    
    return mfcc

# Генератор представительных данных для квантования
def representative_data_gen():
    
    dataset_path = '/home/egor/safeVision/code/Snoring Dataset/1'
    files = os.listdir(dataset_path)
    for file_name in files:
        if file_name.endswith('.wav'):
            audio_path = os.path.join(dataset_path, file_name)
            audio, sr = librosa.load(audio_path, sr=SAMPLING_RATE)
            mfcc = librosa.feature.mfcc(y=audio, n_mfcc=MFCC_NUM, sr=SAMPLING_RATE)
            if mfcc.shape[1] < MFCC_MAX_LEN:
                pad_width = MFCC_MAX_LEN - mfcc.shape[1]
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :MFCC_MAX_LEN]
            mfcc = np.expand_dims(mfcc, axis=-1)
            mfcc = np.expand_dims(mfcc, axis=0)
            if mfcc.shape == (1, MFCC_NUM, MFCC_MAX_LEN, 1):
                yield [mfcc.astype(np.float32)]
            else:
                print(f"Неверный размер тензора для файла {file_name}: {mfcc.shape}")


# Настройка и конвертация модели
converter = tf.lite.TFLiteConverter.from_keras_model(model) 
converter.optimizations = [tf.lite.Optimize.DEFAULT] 
converter.experimental_new_quantizer = True
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] 
converter.inference_input_type = tf.int8  # или tf.uint8
converter.inference_output_type = tf.int8  # или tf.uint8

# Конвертация и сохранение модели
tflite_quant_model = converter.convert() 
with open('quant_28_128_breath_3.tflite', 'wb') as f: 
    f.write(tflite_quant_model)

print("Модель успешно конвертирована и сохранена как quant_model.tflite")
