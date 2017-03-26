# -*- coding: utf-8 -*-
import os, glob
import shutil
import logging
from collections import Counter

os.chdir("/Users/euclidesafonso/Documents")
for file in glob.glob(".x**"):

    source_files = (os.path.join("/Users/euclidesafonso/Documents/WordDocuments", file))
    
##--------------------------------DESTINATION-----------------------------------
    
    if file:
        if os.path.isfile(source_files):
            print("Copied")
        else:
    
            print("Error: %s file already exists!")
                  
    destination_folder =('/Users/euclidesafonso/Documents/WordDocuments/PagesDocs')
    shutil.move(source_files,destination_folder)
####    
##-----------------------------------------------------------------------------
##    print(source_files)
    
##    print(os.path.join("/Users/euclidesafonso/documents", file))
    print("File: ------ {}".format(file))
##

    
#----------------------------WORD_DOCUMENTS ------------------------------
##    destination_folder =('/Users/euclidesafonso/Documents/WordDocuments')
##    shutil.copytree('/users/euclidesafonso/tensorflow', '/Volumes/Backup2/{}/'.format(folderName), symlinks=False, ignore=None)


##!/usr/bin/python
##    myfile= source_files
## 
## if file exists, delete it 
##    if os.path.isfile(myfile):
##        os.remove(myfile)
##    else:    ## Show an error ##
##        print("Error: %s file not found" % myfile)
##
