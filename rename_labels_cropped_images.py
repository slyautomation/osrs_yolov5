import os
import pathlib
directory = pathlib.Path(__file__).parent.resolve()
path = '\\runs\\detect\\exp\\crops\\name of class\\'
print(str(directory) + path)


files = os.listdir(str(directory) + path)
#
print(str(files))
os.chdir(str(directory) + path)
import os
for filename in os.listdir(str(directory) + path):
    print(filename)
    os.rename(filename, filename.replace('full_', ''))