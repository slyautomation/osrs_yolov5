import os
import pathlib
import zipfile as z

# project path
directory = pathlib.Path(__file__).parent.resolve()

# dataset path
zipPath = '\\datasets\\'

# final path
dir_name = str(directory) + zipPath

# change directory to ensure files can be copied and unzipped
os.chdir(dir_name)
print(str(directory))

#first command is to change directory in command prompt (cmd.exe)
first_cmd = 'cd ' + str(directory) + '\datasets\\'

#final command is to copy the binary data of any file in the folder ending in cow.zip.xxx and save result as output.zip
final_cmd = 'copy /B cow.zip.* output.zip'

# run commands in command prompt and terminate once done (/c)
os.system('cmd /c ' + first_cmd + ' & dir & ' + final_cmd)

# unzip resulting file output.zip, this will have all the jpg and xml files
with z.ZipFile(str(directory) + "\datasets\output.zip", 'r') as zip_ref:
    zip_ref.extractall(str(directory) + "\datasets\\")
