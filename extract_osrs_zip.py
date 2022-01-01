import os
import pathlib
import zipfile as z
directory = pathlib.Path(__file__).parent.resolve()

zipPath = '\\datasets\\'
dir_name = str(directory) + zipPath
os.chdir(dir_name)
print(str(directory))

first_cmd = 'cd ' + str(directory) + '\datasets\\'
final_cmd = 'copy /B cow.zip.* output.zip'

os.system('cmd /c ' + first_cmd + ' & dir & ' + final_cmd)

with z.ZipFile(str(directory) + "\datasets\output.zip", 'r') as zip_ref:
    zip_ref.extractall(str(directory) + "\datasets\\")
