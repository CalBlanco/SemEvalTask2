from nltk.tokenize import WordPunctTokenizer
import sys
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import pipeline
import sys
import os
import pandas as pd
import torch 


#from data_util import translate_entity, translate_entities


device = "cuda" if torch.cuda.is_available() else "cpu"

ner_tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER") #NER Model
ner_model = AutoModelForTokenClassification.from_pretrained("dslim/distilbert-NER")

nlp = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, device=device)


def merge_entities(ner_output):
    merged_entities = []
    current_entity = None

    for item in ner_output:
        if item['entity'].startswith('B-'):  # Begin a new entity
            if current_entity:
                merged_entities.append(current_entity)
            current_entity = {
                'entity': item['entity'],  # Keep the 'B-' prefix
                'words': [item['word']],
                'start_index': item['index'],
                'end_index': item['index']
            }
        elif item['entity'].startswith('I-') and current_entity:
            if item['entity'][2:] == current_entity['entity'][2:]:  # Match base entity type
                current_entity['words'].append(item['word'])
                current_entity['end_index'] = item['index']
        else:  # If not part of a contiguous entity, finish current entity
            if current_entity:
                merged_entities.append(current_entity)
                current_entity = None

    if current_entity:  # Append any remaining entity
        merged_entities.append(current_entity)

    # Convert merged entities into desired format
    result = [
        (" ".join(entity['words']), entity['entity'])
        for entity in merged_entities
    ]
    return result




model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt") 
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX") #Translator Model


model.to(device)


tr_table = { #Translation table to take in our expected language names and use them for facebooks model
    'ar': 'ar_AR',
    'de': 'de_DE',
    'es': 'es_XX',
    'fr': 'fr_XX',
    'it': 'it_IT',
    'ja': 'ja_XX'
}

def translate(source, lang): #translate wrapper 
    lang = tr_table[lang]
    inputs = tokenizer(source, return_tensors="pt", padding=True, truncation=True, max_length=200).to(device)
    generated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[lang])
    return tokenizer.batch_decode(generated, skip_special_tokens=True)



def mask_one(source:str, langs:list[str]): #simple masking function to place the translated token into the pre-translated sentence
    res = nlp(source) #get NER tags
    merged = merge_entities(res) #Merge 
    print(merged)

    translations = {}
    for ent in merged:
        #tr = translate_entity(ent[0], langs) #get our translated version of the entity if possible 
        tr = []
        if tr != []: #don't place nothing in for the translation
            translations[ent[0]] = tr
        

    
    updated = []
    for token in WordPunctTokenizer().tokenize(source):
        try:
            updated.append(translations[token][0])
        except:
            updated.append(token)

    return ' '.join(updated)



# Need to make jsons with the form
# Predictions Format
# {
#   "id": "Q627784_0",
#   "prediction": "Come viene ricordato e onorato Yu il Grande nella storia e cultura cinese di oggi?",
# }



def load_testing_data(path:str):
    """Load in all XC Testing data"""
    data = {}

    path = os.path.join(os.path.dirname(__file__), path) #fix path

    for lang in os.listdir(path): #iterate over the files to load in dfs
        file = f'{path}/{lang}/test.jsonl'
        frame = pd.read_json(file, lines=True)
        data[lang] = frame

    return data 

from tqdm import tqdm
import json

def run_test(name:str, data_path:str, mask_fn=lambda x: x, batch_size=32):
    """Run test over the XC dataset

    Args:
        - name (str): Name for the test
        - mask_fn (fn(str) -> str): a Masking function for the source sentence [Default returns itself]
    """

    data = load_testing_data(data_path)
    test_path = os.path.join(os.path.dirname(__file__), f'test_results/{name}')

    try:
        os.makedirs(test_path)
    except FileExistsError:
        print('Path already exists')
        
    for lang, df in data.items(): #iterate over all dataframes

        masked_sources = [] #Get masked version of sentences
        row_ds = []
        for i, row in  df.iterrows():
            masked_source = mask_fn(row['source'])
            masked_sources.append(masked_source)
            row_ds.append(row['id'])

        size = len(masked_sources) #size for determining batch count
        
        pairs = []
        for batch in tqdm(range(round(size/batch_size)), desc=f'Translating batches for {lang}'): #batch 
            start = batch*batch_size
            end = (batch+1) * batch_size if (batch+1) * batch_size < size else size

            
            sources = masked_sources[start:end]
            #print(len(sources))
            translated = translate(sources, lang)
            #print(len(translated))

            id_translation_pairs = list(zip(row_ds[start:end], translated))
            pairs.extend(id_translation_pairs)

        
        with open(f'{test_path}/{lang}.jsonl', 'w') as f:
            shit = [json.dumps({'id': id, 'prediction': prediction})+"\n" for id, prediction in pairs]
            f.writelines(shit)





if __name__ == "__main__":
    data_path = sys.argv[1]
    test_name = sys.argv[2]

    print(f"\nStarting test - {test_name} - at path {data_path}\n {'='*20}\n")
    run_test(test_name, data_path)



