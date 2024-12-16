import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

# Параметры записи
SAMPLE_RATE = 16000  # Частота дискретизации, 16 кГц
DURATION = 1  # Длительность записи (1 секунда)
AUDIO_PATH = 'temp_audio.wav'  # Временный файл для хранения аудиофрагмента

def record_audio():
    # Записываем звук с микрофона
    print("Start recording...")
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()  # Ожидаем завершения записи
    print("End recording.")
    write(AUDIO_PATH, SAMPLE_RATE, audio_data)  # Сохраняем во временный файл
    return AUDIO_PATH
