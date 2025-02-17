# Author: Maj Ferlan Kovič

import os
import json

def checkQA(directory: str):

    noAnser = []
    noQustion = []
    for root, dir, file in os.walk(directory):
        if not dir and file:
            QAs = [item for item in file if '.json' in item]
            for jsonData in QAs:
                path = root + '\\' + jsonData
                with open(path, 'r') as file:
                    data = json.load(file)
                    for QandA in data:
                        if QandA['Q'] == '':
                            noQustion.append(path)
                        if QandA['A'] == '':
                            noAnser.append(path)
    print('No answer provided:\n')
    for a in noAnser:
        print(a)
    print()
    print('No question provided:\n')
    for q in noQustion:
        print(q)
