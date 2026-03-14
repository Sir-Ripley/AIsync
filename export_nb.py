import json

with open('BigTest.ipynb', 'r') as f:
    nb = json.load(f)

with open('BigTest.py', 'w') as f:
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            f.write(''.join(cell['source']) + '\n')
