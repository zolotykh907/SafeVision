import threading
import time
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tensorflow.keras.models import load_model
import noisereduce as nr
from queue import Queue

from predict_1s import predict_from_file

MODEL_PATH = r'D:\python\safevision\SafeVision\breath detection\models\h5\model_dataset2.h5'
model = load_model(MODEL_PATH)

duration = 1
sample_rate = 7000
buffer_size = sample_rate * duration

audio_buffer = np.zeros(buffer_size)

# Ограниченная очередь для задач обработки
queue_size = 10  # Максимальное количество задач в очереди
task_queue = Queue(maxsize=queue_size)

# Глобальная переменная для управления цветом графика
prediction_high = False

def record_audio():
    def callback(indata, frames, time, status):
        if status:
            print(status)
        global audio_buffer
        audio_buffer = np.roll(audio_buffer, -frames)
        audio_buffer[-frames:] = indata[:, 0]

    with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate):
        while True:
            time.sleep(1)

def update_plot(frame):
    global audio_buffer, prediction_high, line
    line.set_ydata(audio_buffer)
    
    # Изменяем цвет графика в зависимости от предсказания
    if prediction_high:
        line.set_color('green')
    else:
        line.set_color('blue')
    return line,

def process_audio():
    global audio_buffer
    while True:
        time.sleep(duration)

        # Копируем буфер для обработки
        audio_copy = audio_buffer.copy()

        # Попытка добавить задачу в очередь
        try:
            task_queue.put_nowait(audio_copy)
        except Queue.Full:
            print("Очередь переполнена, пропускаем обработку текущего буфера.")

def worker():
    global prediction_high
    while True:
        # Забираем задачу из очереди
        audio_copy = task_queue.get()

        # Уменьшаем шум (при необходимости)
        #audio_copy = nr.reduce_noise(y=audio_copy, sr=sample_rate)

        # Выполняем предсказание
        pred = predict_from_file(audio_copy, model)
        print(f"Prediction: {pred}")

        # Обновляем состояние глобальной переменной
        prediction_high = pred > 0.5

        # Сообщаем, что задача выполнена
        task_queue.task_done()

# Настройка графика
fig, ax = plt.subplots()
ax.set_ylim(-1, 1)
ax.set_xlim(0, buffer_size / sample_rate)
ax.set_xlabel('Время (сек)')
ax.set_ylabel('Амплитуда')
line, = ax.plot(np.arange(buffer_size) / sample_rate, audio_buffer, color='blue')

# Запуск потока записи
record_thread = threading.Thread(target=record_audio, daemon=True)
record_thread.start()

# Запуск потока обработки
process_thread = threading.Thread(target=process_audio, daemon=True)
process_thread.start()

# Запуск рабочего потока для очереди
worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()

# Анимация графика
ani = FuncAnimation(fig, update_plot, interval=1000, blit=True)
plt.show()
