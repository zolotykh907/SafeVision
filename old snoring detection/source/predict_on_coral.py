import tensorflow as tf
import numpy as np
from scipy import signal
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
import time 

# Путь к модели и аудиофайлу
MODEL_PATH = 'Igor_last.tflite'
AUDIO_PATH = '1_1.wav'

# Функция загрузки и ресемплирования аудиофайла до 16kHz моно
def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)

    # Проверяем и пересэмплируем, если sample_rate не равен 16000
    if sample_rate != 16000:
        wav = resample_wav(wav, sample_rate)

    return wav

def resample_wav(wav, sample_rate):
    num_samples = int(tf.shape(wav)[0] * 16000 / sample_rate)
    wav_numpy = wav.numpy()
    wav_resampled = signal.resample(wav_numpy, num_samples)
    return tf.convert_to_tensor(wav_resampled, dtype=tf.float32)

# Преобразование в спектрограмму и приведение к uint8
def preprocess_audio(file_path, target_size):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:16000]  # Обрезаем или дополняем до 1 секунды (16,000 samples)
    zero_padding = tf.zeros([16000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)

    # Создаем спектрограмму
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)  # Добавляем канал [time, freq, 1]

    # Приводим спектрограмму к нужному размеру и типу uint8
    return spectrogram

# Загрузка модели и настройка для Coral
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()

# Получаем размерность входа модели
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
scale, z_point = input_details[0]['quantization']
target_size = (input_shape[1], input_shape[2])  # [высота, ширина]

# Обрабатываем аудиофайл
spectrogram = preprocess_audio(AUDIO_PATH, target_size)
spectrogram = spectrogram.numpy()
spectrogram = ((spectrogram/scale)+z_point).astype(np.int8)
start = time.time()
# Подаём спектрограмму в модель

interpreter.set_tensor(input_details[0]['index'], [spectrogram])
interpreter.invoke()
scale0, z_point0 = output_details[0]['quantization']
# Получаем предсказание
output_data = (interpreter.get_tensor(output_details[0]['index']).astype('float32') - z_point0) * scale0
prediction = output_data#np.argmax(output_data)
print(time.time() - start)

# Выводим результат
print("Предсказание:", prediction)
