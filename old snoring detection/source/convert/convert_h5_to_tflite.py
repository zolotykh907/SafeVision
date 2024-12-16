import tensorflow as tf
import numpy as np
import os
from scipy import signal

# Загрузка вашей обученной модели
MODEL_PATH = '/home/egor/safeVision/code/model/my_h5_model_60mb_10e.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Функция для пересэмплирования, если частота не равна 16000
def resample_wav(wav, sample_rate):
    num_samples = int(tf.shape(wav)[0] * 16000 / sample_rate)
    wav_numpy = wav.numpy()
    wav_resampled = signal.resample(wav_numpy, num_samples)
    return tf.convert_to_tensor(wav_resampled, dtype=tf.float32)

# Функция для загрузки и обработки WAV файлов
def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)

    if sample_rate != 16000:
        wav = resample_wav(wav, sample_rate)

    return wav

# Функция для создания спектрограммы
def create_spectrogram(wav, frame_length=320, frame_step=32):
    spectrogram = tf.signal.stft(wav, frame_length=frame_length, frame_step=frame_step)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)  # Добавляем размерность канала
    return spectrogram

# Генератор представительных данных
def representative_data_gen():
    dataset_path = '/home/egor/safeVision/code/Snoring Dataset/1'
    files = os.listdir(dataset_path)
    for file_name in files:
        if file_name.endswith('.wav'):
            audio_path = os.path.join(dataset_path, file_name)
            wav = load_wav_16k_mono(audio_path)
            wav = wav[:16000]  # Ограничиваем до 1 секунды (16,000 семплов)
            spectrogram = create_spectrogram(wav)  # Преобразуем в спектрограмму
            spectrogram = tf.expand_dims(spectrogram, axis=0)  # Добавляем размерность для батча
            yield [spectrogram.numpy().astype(np.float32)]

# Настройка конвертера и квантизации
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_quantizer = True
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Конвертация и сохранение модели
tflite_quant_model = converter.convert()
with open('zalupa.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("Модель успешно конвертирована в TFLite с int8 квантизацией.")
