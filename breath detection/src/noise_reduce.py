import numpy as np
import soundfile as sf
import noisereduce as nr
import librosa

def reduce_noise_and_save(input_file, output_file, target_sample_rate):
    # Загрузка аудиофайла
    audio_data, original_sample_rate = sf.read(input_file)

    # Проверка, нужна ли передискретизация
    if original_sample_rate != target_sample_rate:
        print(f"Передискретизация аудиоданных с {original_sample_rate} Гц до {target_sample_rate} Гц...")
        audio_data = librosa.resample(audio_data.T, orig_sr=original_sample_rate, target_sr=target_sample_rate).T

    # Применение шумоподавления
    reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=target_sample_rate)

    # Сохранение аудиофайла с шумоподавлением
    sf.write(output_file, reduced_noise_audio, target_sample_rate)

    print(f"Аудиофайл с шумоподавлением сохранен: {output_file}")

if __name__ == "__main__":
    # Запрос у пользователя пути к входному аудиофайлу
    input_file = r'D:\python\safevision\Respiratory_Sound_Database\PROCESSING_BIG_DATASET\101_1b1_Al_sc_Meditron_processed.wav'
    

    # Запрос у пользователя пути к выходному аудиофайлу
    output_file = r'test.wav'

    # Запрос у пользователя частоты дискретизации
    target_sample_rate = 7000

    # Вызов функции для применения шумоподавления и сохранения результата
    reduce_noise_and_save(input_file, output_file, target_sample_rate)