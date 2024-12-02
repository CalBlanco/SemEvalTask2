from data_util import retrieve_data, translate_entities
from named_entity_recognition import ner_baseline
from database import query_by_name, query_by_instance, query_by_id
from datasets import TagingDataset
from gru_model import GRUModel
from torch.utils.data import DataLoader
from collaters import Collator
from data_util import retrieve_coNER, iob_to_entities, ner_accuracy
from sklearn.model_selection import train_test_split

#Uncomment this block to train GRU
""" 

BATCH_SIZE = 64

x_train, y_train = retrieve_coNER("en_coNER_train.json")
x_val, y_val = retrieve_coNER("en_coNER_validation.json")

x_val, x_test, y_val, y_test = train_test_split(x_val,y_val, test_size=0.5, random_state=64)

train_dataset = TagingDataset(x_train,y_train, training=True)
val_dataset = TagingDataset(x_val, y_val, token_vocab=train_dataset.token_vocab, tag_vocab=train_dataset.tag_vocab, training=False)
test_dataset = TagingDataset(x_test, y_test, token_vocab=train_dataset.token_vocab, tag_vocab=train_dataset.tag_vocab, training=False)

collater = Collator(train_dataset.token_vocab, train_dataset.tag_vocab)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collater.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collater.collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collater.collate_fn)

gru = GRUModel(train_dataset.token_vocab, train_dataset.tag_vocab)

gru.fit(train_loader, val_loader, epochs=1)

out = gru.predict(test_loader)

decode = gru.decode(out)

gru_test = GRUModel(train_dataset.token_vocab, train_dataset.tag_vocab)
gru_test.load_model("best_model.pt")

out_test = gru_test.predict(test_loader)

decode_test = gru_test.decode(out_test)

pred_entities = iob_to_entities(decode_test)
true_entities = iob_to_entities(list(zip(x_test, y_test)))

print(ner_accuracy(pred_entities, true_entities)) """



#Uncomment this block to see translate_entities in action
""" 
example = [[('North America', 'Thing'), ('John', 'Person')],[('North America', 'Thing')]]

ents = translate_entities(example)
print(ents) 
"""