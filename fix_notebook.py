import json

with open('c:/Users/jaysi/Desktop/Desktop/Ai-engineering/notebook/week2/03hybridSearch.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'embeddings=embeddings' in source:
            cell['source'] = ['embeddings = get_embedding_batch(text_to_embed)\n']
        
        if 'pointstructs=[]' in source:
            new_source = []
            for line in cell['source']:
                line = line.replace('"text-embedding-3-small"', '"text-embedding-model-3-small"')
                new_source.append(line)
            cell['source'] = new_source

with open('c:/Users/jaysi/Desktop/Desktop/Ai-engineering/notebook/week2/03hybridSearch.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
