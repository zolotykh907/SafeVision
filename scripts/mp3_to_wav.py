import os
from pydub import AudioSegment

INPUT_DIRECTORY = '/home/egor/safeVision/SafeVision/breath detection/Breath_Dataset/vidoh'

files = os.listdir(INPUT_DIRECTORY)

for filename in files:
    if filename.endswith('.mp3'):
        mp3_path = os.path.join(INPUT_DIRECTORY, filename)

        audio = AudioSegment.from_mp3(mp3_path)

        wav_filename = filename.replace('.mp3', '.wav')
        wav_path = os.path.join(INPUT_DIRECTORY, wav_filename)

        audio.export(wav_path, format='wav')
        
        print(f'Converted {filename} to {wav_filename}')