from nltk.tokenize import WordPunctTokenizer
import sys
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import pipeline
import sys
import os
import pandas as pd
import torch 
from database import query_by_alias, query_by_name
import json

# ARGS python huggingface.py <xc_data_path> <test_name> <gpu #>
# This Code was used initially to run the pipeline on the school compute bc the school compute does not use github version control tracking was made difficult
# 

gpu = sys.argv[3] if len(sys.argv) > 2 else "0" #default to first gpu 
device = f"cuda:{sys.argv[3]}" if torch.cuda.is_available() else "cpu"

ner_tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER") #NER Model
ner_model = AutoModelForTokenClassification.from_pretrained("dslim/distilbert-NER")

nlp = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, device=device, aggregation_strategy="average")


def merge_entities(ner_output):
    result = []

    for item in ner_output:
        name = item['word']
        ent_type = item['entity_group']

        result.append((name, ent_type))

    
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
    #print(merged)

    translations = {}
    for ent in merged:
        #tr = translate_entity(ent[0], langs) #get our translated version of the entity if possible 
        q = query_by_alias(ent[0])
        try:
            name = query_by_name(q[0], ','.join(langs))
            tr = name
            translations[ent[0]] = tr
        except:
            pass
        

    
    updated = []
    for token in WordPunctTokenizer().tokenize(source):
        try:
            updated.append(translations[token][0])
        except:
            updated.append(token)

    return ' '.join(updated)

def concat_mask(source:str, lang:str):
    res = nlp(source) #get entities 
    merged = merge_entities(res)

    translated = []

    for ent in merged:
        ent_name = ent[0]
        try:
            q = query_by_name(ent_name, projections=lang) #try to get by name
            if len(q) == 0:
                q = query_by_alias(ent_name) #try to get by alias
                q = query_by_name(q[0][0], projections=lang)
            

            ent_translated = q[0][0]
            translated.append(f'<{lang}> {ent_translated} </{lang}>')

        except:
            pass
        
    source = source + ' ' + ' '.join(translated)

    return source 

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

def run_test(name:str, data_path:str, mask_fn=lambda x,y: x, batch_size=32):
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
            masked_source = mask_fn(row['source'],[lang])
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
    run_test(test_name, data_path, mask_fn=concat_mask)
    
    