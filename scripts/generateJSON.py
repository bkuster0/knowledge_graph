# Author: Maj Ferlan Kovič
import os
import json

for path, subdirs, files in os.walk('.'):

    data = []
    dictQA = {}
    dictQA['Q'] = ''
    dictQA['A'] = ''
    data.append(dictQA)

    for name in files:
        names = os.path.join(path, name)
        # za vsak slučaj (ni potrebno)
        if 'BACKUP' in names:
            continue
        tmp = name.split('.')[0]
        if len(tmp) == 4:
            namePathJSON = os.path.join(path, tmp + '_qa.json')
            with open(namePathJSON, 'w') as fp:
                json.dump(data, fp, indent=4)
