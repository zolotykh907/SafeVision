import threading
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import noisereduce as nr
import tensorflow as tf

# Путь к модели TFLite
MODEL_PATH = r'D:\python\safevision\SafeVision\breath detection\models\tflite\model_for_BIG_DATASET.tflite'
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Получаем входной и выходной тензоры
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Константы для записи звука
duration = 1
sample_rate = 44100
buffer_size = sample_rate * duration

audio_buffer = np.zeros(buffer_size)
data_ready_event = threading.Event()

def record_audio():
    def callback(indata, frames, time, status):
        if status:
            print(status)
        global audio_buffer
        audio_buffer[:] = indata[:, 0]
        data_ready_event.set()

    with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate, blocksize=buffer_size):
        while True:
            data_ready_event.wait()
            data_ready_event.clear()

def update_plot(frame):
    global audio_buffer
    line.set_ydata(audio_buffer)
    return [line]

def process_audio():
    global audio_buffer
    while True:
        data_ready_event.wait()  # Ждём сигнал о готовности данных
        audio_copy = audio_buffer.copy()
        data_ready_event.clear()

        # Уменьшение шума
        audio_copy = nr.reduce_noise(y=audio_copy, sr=sample_rate)

        # Подготовка данных для модели
        input_data = np.expand_dims(audio_copy, axis=0).astype(np.float32)

        # Выполнение предсказания
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Печать результата
        print("Prediction:", output_data)

fig, ax = plt.subplots()
ax.set_ylim(-1, 1)
ax.set_xlim(0, buffer_size / sample_rate)
ax.set_xlabel('Время (сек)')
ax.set_ylabel('Амплитуда')
line, = ax.plot(np.arange(buffer_size) / sample_rate, audio_buffer)

record_thread = threading.Thread(target=record_audio)
record_thread.daemon = True
record_thread.start()

process_thread = threading.Thread(target=process_audio)
process_thread.daemon = True
process_thread.start()

ani = FuncAnimation(fig, update_plot, interval=100, blit=True)
plt.show()
