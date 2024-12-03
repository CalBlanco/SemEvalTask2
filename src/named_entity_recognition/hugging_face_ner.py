from data_util import retrieve_coNER
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

def ner_baseline(file_path: str, num_samples = 100000)->list[tuple]:
    """NER Baseline model from Huggingface

    ARGS
        **file_path**   -- path to the file containing the data
        **num_samples** -- number of samples to be processed from the first num_samples of the retrieved data

    RETURNS
        A list of tuples, where each tuple contains a list of predicted entities and a list of true entity wikidata ids.
    """
    tokens, ner_tags = retrieve_coNER(file_path)
    data = list(zip(tokens, ner_tags))

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    data_ner = []
    count = 0
    for source, ner_tags in data:
        source = " ".join(source)
        print(source)
        ner_results = nlp(source)
        entities_found = []
        first_entity = True
        named_entity = ""
        for data in ner_results:
            print(data)
            if data['entity'][0] == 'B':
                if first_entity == False:
                    entities_found.append((named_entity[:-1], tag))
                    named_entity = ""
                tag = data['entity'][2:]
                first_entity = False
            named_entity += data['word']
            named_entity += " "
        entities_found.append((named_entity[:-1], data['entity'][2:]))


        # Remove # from entities which the model puts there for some reason
        clean_entities_found = []
        for entity, tag in entities_found:
            if "#" not in source:
                entity = entity.replace(" #", "")
                entity = entity.replace("#", "")
            clean_entities_found.append((entity, tag))

        data_ner.append((clean_entities_found)) # append predicted entities
        count += 1
        if count == num_samples:
            break
    return data_ner

# example usage
#print(ner_baseline('en_coNER_train.json', 20))
