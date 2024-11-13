from data_util import retrieve_data
from named_entity_recognition import ner_baseline
from database import query_by_name, query_by_instance, query_by_id


out = ner_baseline("fr", num_samples=10) #generate NER baseline results 
print(out)

ids = [] #Create a list of just the ids we should have recieved
for item in out:
    ids.extend(item[1])

print(query_by_id(ids, 'wiki_id, name, instance_of, fr')) #print out query results
