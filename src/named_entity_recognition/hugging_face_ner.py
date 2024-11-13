import sys
import os
from data_util import retrieve_data
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

def ner_baseline(lang, num_samples = 100000, multi_lang = False)->list[tuple]:
    """NER Baseline model from Huggingface

    ARGS
        **lang**        -- language of the data to be retrieved, if multi_lang is True, lang should be a list of languages
        **num_samples** -- number of samples to be processed from the first num_samples of the retrieved data
        **multi_lang**  --  if True, the data is retrieved from multiple languages training data
    
    RETURNS
        A list of tuples, where each tuple contains a list of predicted entities and a list of true entity wikidata ids.
    """
    sources = []
    entities = []
    if multi_lang:
        for language in lang:
            sources_lang, entities_lang = retrieve_data(language, target=False)
            sources.extend(sources_lang)
            entities.extend(entities_lang)
    else:
        sources, entities = retrieve_data(lang, target=False)
    data = list(zip(sources, entities))

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    data_ner = []
    count = 0
    for source, entity_list in data:
        ner_results = nlp(source)
        entities_found = []
        first_entity = True
        named_entity = ""
        for data in ner_results:
            if data['entity'][0] == 'B':
                if first_entity == False:
                    entities_found.append(named_entity[:-1])
                    named_entity = ""
                first_entity = False
            named_entity += data['word']
            named_entity += " "
        entities_found.append(named_entity[:-1])

        # Remove # from entities which the model puts there for some reason
        clean_entities_found = []
        for entity in entities_found:
            if "#" not in source:
                if "#" in entity:
                    entity = entity.replace(" #", "")
                entity = entity.replace("#", "")
            clean_entities_found.append(entity)

        data_ner.append((clean_entities_found, entity_list)) # append predicted entities and true entity wikidata ids
        count += 1
        if count == num_samples:
            break
    return data_ner

# example usage
# print(ner_baseline(['ja', 'de'], 20, multi_lang=True))