# Author: Maj Ferlan Kovič
import os
import re

def shiftData(path: str, shiftAmount: int):
    # rename in the reverse to avoide file conflicts
    for file in os.listdir(path)[::-1]:
        newName = ''
        toAdd = ''
        splitFile = re.split(r'\.|_', file, 1)
        movedName = '{:04d}'.format(int(splitFile[0]) + shiftAmount)
        if len(splitFile[1]) <= 3:
            toAdd = '.'
        else:
            toAdd = '_'
        newName = movedName + toAdd + splitFile[1]
        os.rename(path + '/' + file, path + '/' + newName)
