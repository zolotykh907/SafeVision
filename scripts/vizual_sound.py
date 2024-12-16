from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt

audio = AudioSegment.from_file("D:\python\safevision\SafeVision\breath detection\Breath_Dataset/vidoh/3_0.mp3")

audio = audio.set_channels(1)

samples = np.array(audio.get_array_of_samples())

duration_in_seconds = len(audio) / 1000
time = np.linspace(0, duration_in_seconds, num=len(samples))

plt.figure(figsize=(10, 5))
plt.plot(time, samples)
plt.title("Звуковая волна")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")
plt.grid()
plt.show()