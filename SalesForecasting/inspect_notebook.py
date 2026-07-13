import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

notebook_path = Path(r"c:\Users\kanak\Downloads\SalesForecasting_Phase1\SalesForecasting\analysis.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total Cells: {len(nb['cells'])}")
print("=" * 80)
for idx, cell in enumerate(nb['cells']):
    cell_type = cell['cell_type']
    exec_count = cell.get('execution_count', 'N/A')
    source = "".join(cell.get('source', []))
    print(f"Cell {idx} - Type: {cell_type} - Exec Count: {exec_count}")
    print("Source (First 400 chars):")
    print(source[:400])
    if 'outputs' in cell and cell['outputs']:
        print(f"Has Outputs: {len(cell['outputs'])} items")
        for out in cell['outputs']:
            if out.get('output_type') == 'stream':
                print("Stream output:", "".join(out.get('text', []))[:150])
            elif out.get('output_type') == 'execute_result':
                print("Execute result:", str(out.get('data', {}).get('text/plain', ''))[:150])
            elif out.get('output_type') == 'error':
                print("ERROR:", out.get('ename'), out.get('evalue'))
    else:
        print("No Outputs")
    print("-" * 80)
