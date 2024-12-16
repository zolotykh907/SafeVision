import os
import librosa
import soundfile as sf

def split_audio_into_segments(input_file, output_dir, segment_duration=1.0, step_duration=0.5):
    audio_data, sample_rate = librosa.load(input_file, sr=None)

    segment_length = int(segment_duration * sample_rate)
    step_length = int(step_duration * sample_rate)

    segments = []
    for start in range(0, len(audio_data) - segment_length + 1, step_length):
        end = start + segment_length
        segment = audio_data[start:end]
        segments.append(segment)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    for i, segment in enumerate(segments):
        output_file = os.path.join(output_dir, f"{base_name}_segment_{i}.wav")
        sf.write(output_file, segment, sample_rate)
        print(f"Сохранен сегмент: {output_file}")

def process_directory(input_dir, output_dir, segment_duration=1.0, step_duration=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".mp3"):
            input_file = os.path.join(input_dir, file_name)
            split_audio_into_segments(input_file, output_dir, segment_duration, step_duration)

if __name__ == "__main__":
    input_dir = r'D:\python\safevision\Respiratory_Sound_Database\sounds_for_cute\new/'

    output_dir = r'D:\python\safevision\Respiratory_Sound_Database\cute_sounds'

    process_directory(input_dir, output_dir)