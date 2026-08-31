import json

with open('../a2-models/assignment2_evaluation.ipynb', 'r') as f:
    nb = json.load(f)

code = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'plt.show()' in source:
            source = source.replace('plt.show()', '')
        if 'shap.initjs()' in source:
            source = source.replace('shap.initjs()', '')
        if 'display(' in source:
            source = source.replace('display(', 'print(')
        code.append(source)

full_code = '\n'.join(code)
with open('exec_nb.py', 'w') as f:
    f.write(full_code)
