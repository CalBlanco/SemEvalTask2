import json
import os
from nltk.tokenize import WordPunctTokenizer


path = os.path.join(os.path.dirname(__file__), '../../data/train/')
path_test = os.path.join(os.path.dirname(__file__), '../../data/XC_test_data/')

def retrieve_data(lang:str, target = True, entity = True, test = False)->tuple[list,list,list]:
    """Retrieve data from a language
    
    ARGS:
        lang   -- String of a desired language (look at data/train/ for language names)
        target -- boolean flag for returning targets for a utterance
        entity -- boolean flag to return entities for a utterance
    """
    if test:
        source_file = os.path.join(path_test, f'{lang}/test.jsonl')
    else:
        source_file = os.path.join(path, f'{lang}/train.jsonl')
    sources = []
    targets = []
    entities = []
    with open(source_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            sources.append(WordPunctTokenizer().tokenize(data['source'].lower()))
            if test:
                targets.append(WordPunctTokenizer().tokenize(data['targets'][0]['translation'].lower()))
            else:
                targets.append(WordPunctTokenizer().tokenize(data['target'].lower()))
            if test:
                entities.append(data['targets'][0]['mention'])
            else:
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