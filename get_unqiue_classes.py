import os
import pathlib

nc = 0  # number of classes
names = []  # class names


def get_unique_classes():
    global nc, names
    path = "\\datasets\\osrs\\"  # dataset root dir
    directory = pathlib.Path(__file__).parent.resolve()
    print(str(directory) + path)
    files = os.listdir(str(directory) + path)
    # print(str(files))
    os.chdir(str(directory) + path)
    for filename in os.listdir(str(directory) + path):
        if filename.endswith('txt'):
            print(filename)
            with open(filename, 'r') as f:
                for line in f:
                    # print(line.find(':'))
                    name_int = line.find(':')
                    name_ent = line[:name_int]
                    print(name_ent)
                    if name_ent not in names:
                        names.append(name_ent)
                        nc += 1
                    else:
                        # 👇️ this runs
                        print('The specified value is already in the list')


get_unique_classes()
print(names)
print(nc)
