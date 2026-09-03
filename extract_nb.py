import json

with open('assignment2_evaluation.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        print(f"--- CELL {i} OUTPUT ---")
        for out in cell.get('outputs', []):
            if out['output_type'] == 'stream':
                print("".join(out['text']))
            elif out['output_type'] == 'execute_result':
                print("".join(out['data'].get('text/plain', [])))
            elif out['output_type'] == 'display_data':
                print("".join(out['data'].get('text/plain', [])))
        print("\n")
