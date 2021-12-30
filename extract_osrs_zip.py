import os

import zipfile

zipPath = './datasets/'

with zipfile.ZipFile(zipPath + 'osrs.zip*', 'r') as zip_ref:
    zip_ref.extractall('./datasets/osrs/')
