from data_util import retrieve_data
from named_entity_recognition import ner_baseline
from database import command, query, query_by_name, query_by_instance

# print(command('PRAGMA table_info(entity_translation)')) view columns of db

# print(query('name, ar, fr', 'WHERE instance_of = "human" AND name LIKE "la%"')) returns the (name, arabic_translation, french_translation) for humans that have a name la* where * represents any amount of anycharacters 

print('Query by instance\n')
#query 5 humans from the table and get their ids, names, and spanish translations
print(query_by_instance("human", "wiki_id, name, es", "LIMIT 5")) 
print('\n\nQuery by Name')
# query for names starting in La, getting their id, name, instance, german, spanish and japanese translation (only return 4)
print(query_by_name("La%", 'wiki_id, name, instance_of, de, es, ja', 'LIMIT 4')) 