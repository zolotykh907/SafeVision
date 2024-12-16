import os

START_NUM = 1330
BEGIN_NAME = '0_'
INPUT_DIRECTORY = r'D:\python\safevision\Respiratory_Sound_Database\cute_sounds/'

files = os.listdir(INPUT_DIRECTORY)

files = [f for f in files if os.path.isfile(os.path.join(INPUT_DIRECTORY, f))]

for index, file in enumerate(files):
    file_name, file_extension = os.path.splitext(file)
    
    new_name = f"{BEGIN_NAME}{START_NUM + index}{file_extension}"
    
    old_path = os.path.join(INPUT_DIRECTORY, file)
    new_path = os.path.join(INPUT_DIRECTORY, new_name)
    
    os.rename(old_path, new_path)
    
    print(f"File {file} renamed to {new_name}")
