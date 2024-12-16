import threading
import time
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

duration = 10
sample_rate = 44100 
buffer_size = sample_rate * duration

audio_buffer = np.zeros(buffer_size)

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
    global audio_buffer
    line.set_ydata(audio_buffer)
    return line,

fig, ax = plt.subplots()
ax.set_ylim(-1, 1)
ax.set_xlim(0, buffer_size / sample_rate)
ax.set_xlabel('Время (сек)')
ax.set_ylabel('Амплитуда')
line, = ax.plot(np.arange(buffer_size) / sample_rate, audio_buffer)

record_thread = threading.Thread(target=record_audio)
record_thread.daemon = True
record_thread.start()

ani = FuncAnimation(fig, update_plot, interval=1000, blit=True)
plt.show()