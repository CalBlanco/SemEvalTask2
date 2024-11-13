from data_util import retrieve_data
from named_entity_recognition import ner_baseline
from database import query_by_name, query_by_instance, query_by_id


print(query_by_id(['Q49'], 'name, instance_of'))

