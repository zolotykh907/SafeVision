import os
from tqdm import tqdm
from pydub import AudioSegment

INPUT_PATH = r'D:\python\safevision\Respiratory_Sound_Database\PROCESSING_BIG_DATASET/'
OUTPUT_PATH = r'D:\python\safevision\Respiratory_Sound_Database\BIG_DATASET\1/'

def cut_audio_segments(audio_file, annotation_file, index):

    audio = AudioSegment.from_wav(audio_file)
    
    with open(annotation_file, 'r') as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) >= 2:
            start_time = float(parts[0]) * 1000
            end_time = float(parts[1]) * 1000

            if end_time - start_time >= 1000:
            
                segment = audio[start_time:end_time]
                
                part_index = 0
                for sub_start in range(0, len(segment) - 1000 + 1, 500):  
                    sub_end = sub_start + 1000  
                    if sub_end <= len(segment): 
                        part_segment = segment[sub_start:sub_end]
                        
                        output_path = f"{OUTPUT_PATH}/{index}_segment_{i+1}_part_{part_index + 1}.wav"
                        part_segment.export(output_path, format="wav")
                        #print(f"save: {output_path}")
                        part_index += 1


if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
else:
    print(f"Folder `{OUTPUT_PATH}` already exists")

txt_file_names = [file for file in os.listdir(INPUT_PATH) if file.endswith('.txt')]

index = 0

for txt_file_name in tqdm(txt_file_names, desc='Total'):
    if txt_file_name.endswith('.txt'):
        txt_file_name_path = INPUT_PATH + txt_file_name
        wav_file_name_path = txt_file_name_path.replace('.txt', '.wav')

        cut_audio_segments(wav_file_name_path, txt_file_name_path, index)
        index += 1
    else:
        print("It`s no txt file")

print("FINISH!!!")