import json
with open('assignment2_evaluation.ipynb', 'r') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if line == "df = pd.read_csv('../AML_Book-Data/Data/a1_cleaned_data.csv')\n":
                new_source.append("df = pd.read_csv('./data/a1_cleaned_data.csv')\n")
            else:
                new_source.append(line)
        cell['source'] = new_source
with open('assignment2_evaluation.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
