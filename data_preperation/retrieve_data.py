import json

def retrieve_data(lang, target = True, entity = True):
    source_file = f'data/train/{lang}/train.jsonl'
    sources = []
    targets = []
    entities = []
    with open(source_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            sources.append(data['source'])
            targets.append(data['target'])
            entities.append(data['entities'])
    if not target and not entity:
        return sources
    if not target:
        return sources, entities
    if not entity:
        return sources, targets
    else:
        return sources, targets, entities

# example usage
# sources, targets, entities = retrieve_data('ja')