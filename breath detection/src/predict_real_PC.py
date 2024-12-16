import threading

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tensorflow.keras.models import load_model
import noisereduce as nr

from predict_1s import predict_from_file

MODEL_PATH = r'D:\python\safevision\SafeVision\breath detection\models\h5\model_for_BIG_DATASET.h5'
model = load_model(MODEL_PATH)

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
    return line,

def process_audio():
    global audio_buffer
    while True:
        data_ready_event.wait()
        audio_copy = audio_buffer.copy()
        data_ready_event.clear()

        #audio_copy = nr.reduce_noise(y=audio_copy, sr=sample_rate)

        prediction_thread = threading.Thread(target=predict_from_file, args=(audio_copy, model))
        prediction_thread.start()

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
