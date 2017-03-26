import os, glob
import shutil
import logging
from collections import Counter

def src (source_path, dotExtension):
    os.chdir(source_path)
    for file in glob.glob(dotExtension):
        #print(file)
        print(os.path.join(source_path, file))

    return ("File: ------ {}".format(file))
    

 
