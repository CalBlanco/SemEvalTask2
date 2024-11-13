import json
import os


path = os.path.join(os.path.dirname(__file__), '../../data/train/')

def retrieve_data(lang:str, target = True, entity = True)->tuple[list,list,list]:
    """Retrieve data from a language
    
    ARGS:
        lang   -- String of a desired language (look at data/train/ for language names)
        target -- boolean flag for returning targets for a utterance
        entity -- boolean flag to return entities for a utterance
    """
    source_file = os.path.join(path, f'{lang}/train.jsonl')
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
    elif not target:
        return sources, entities
    elif not entity:
        return sources, targets
    else:
        return sources, targets, entities

# example usage
# sources, targets, entities = retrieve_data('ja')