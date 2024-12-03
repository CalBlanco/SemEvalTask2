from data_util import retrieve_data, translate_entities
from named_entity_recognition import ner_baseline
from database import query_by_name, query_by_instance, query_by_id
from datasets import TagingDataset
from gru_model import GRUModel
from transformer_ner import TransformerNER
from torch.utils.data import DataLoader
from collaters import Collator
from data_util import retrieve_coNER, iob_to_entities, ner_accuracy, class_accuracy
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

#Uncomment this block to train GRU


BATCH_SIZE = 64

x_train, y_train = retrieve_coNER("en_coNER_train.json")
x_val, y_val = retrieve_coNER("en_coNER_validation.json")
#x_train, y_train = retrieve_data("de", target = False)
#x_val, y_val = retrieve_data("de", target = False)

x_val, x_test, y_val, y_test = train_test_split(x_val,y_val, test_size=0.5, random_state=64)

train_dataset = TagingDataset(x_train,y_train, training=True)
val_dataset = TagingDataset(x_val, y_val, token_vocab=train_dataset.token_vocab, tag_vocab=train_dataset.tag_vocab, training=False)
test_dataset = TagingDataset(x_test, y_test, token_vocab=train_dataset.token_vocab, tag_vocab=train_dataset.tag_vocab, training=False)

collater = Collator(train_dataset.token_vocab, train_dataset.tag_vocab)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collater.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collater.collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collater.collate_fn)

"""

# Transformer Model
#transformer = TransformerNER(train_dataset.token_vocab, train_dataset.tag_vocab)

#transformer.fit(train_loader, val_loader, epochs=30)

#out = transformer.predict(test_loader)

#decode = transformer.decode(out, x_test)

transformer_test = TransformerNER(train_dataset.token_vocab, train_dataset.tag_vocab)
accuracy_results_transformer = []
class_accuracy_results_transformer = []
for i in range(16,17):
    transformer_test.load_model(f"../transformer_ner_{i}.pt")

    out_test = transformer_test.predict(test_loader)

    decode_test = transformer_test.decode(out_test, x_test)

    pred_entities = iob_to_entities(decode_test)
    #true_entities = iob_to_entities(list(zip(x_test, y_test)))
    class_acc, entity_types = class_accuracy(pred_entities, y_test)

    correct = []
    incorrect = []

    for class_type in entity_types:
        if entity_types[class_type] > 0:
            if class_type.endswith("_correct"):
                class_name = class_type.replace("_correct", "")
                correct.append((class_name, entity_types[class_type]))
            elif class_type.endswith("_incorrect"):
                class_name = class_type.replace("_incorrect", "")
                incorrect.append((class_name, entity_types[class_type]))
    plt.figure(figsize=(20, 18))  # Make figure larger with width=15, height=10
    if len(entity_types) > 0:
        # Unzip the tuples into separate lists for x and y values
        class_names, counts = zip(*correct)
        incorrect_names, incorrect_counts = zip(*incorrect)
        for class_name in class_names:
            if class_name not in incorrect_names:
                incorrect_names = incorrect_names + (class_name,)
                incorrect_counts = incorrect_counts + (0,)
        for class_name in incorrect_names:
            if class_name not in class_names:
                class_names = class_names + (class_name,)
                counts = counts + (0,)
        # Create x positions for the bars
        x = np.arange(len(class_names))
        width = 0.35  # Width of the bars

        # Create the stacked bars
        plt.bar(x, counts, width, label='Correct')
        plt.bar(x, incorrect_counts, width, bottom=counts, label='Incorrect')

        # Customize the plot
        plt.xticks(x, class_names, rotation=90)
        plt.title(f"TransformerNER_{i} Class Accuracy")
        plt.xlabel("Class Type")
        plt.ylabel("Count")
        plt.legend()
        plt.xticks(rotation=90)
        plt.savefig(f"TransformerNER_{i}_class_accuracy.png")
        plt.close()


accuracy_results_transformer.append(f"TransformerNER_{i} accuracy: {ner_accuracy(pred_entities, y_test)}")
class_accuracy_results_transformer.append(f"TransformerNER_{i} class accuracy: {class_acc}")

for acc in zip(accuracy_results_transformer, class_accuracy_results_transformer):
    print(acc)

_, ner_data = ner_accuracy(pred_entities, y_test)

for acc in ner_data:
    print(acc)
"""

# GRU model
gru = GRUModel(train_dataset.token_vocab, train_dataset.tag_vocab)

gru.fit(train_loader, val_loader, epochs=40)

out = gru.predict(test_loader)


decode = gru.decode(out, x_test)

gru_test = GRUModel(train_dataset.token_vocab, train_dataset.tag_vocab)
accuracy_results_gru = []
class_accuracy_results_gru = []
for i in range(1,41):
    gru_test.load_model(f"gru_ner_{i}.pt")

    out_test = gru_test.predict(test_loader)

    decode_test = gru_test.decode(out_test, x_test)

    pred_entities = iob_to_entities(decode_test)
    true_entities = iob_to_entities(list(zip(x_test, y_test)))
    class_acc, entity_types = class_accuracy(pred_entities, true_entities)

    correct = []
    incorrect = []

    for class_type in entity_types:
        if entity_types[class_type] > 0:
            if class_type.endswith("_correct"):
                class_name = class_type.replace("_correct", "")
                correct.append((class_name, entity_types[class_type]))
            elif class_type.endswith("_incorrect"):
                class_name = class_type.replace("_incorrect", "")
                incorrect.append((class_name, entity_types[class_type]))
    plt.figure(figsize=(20, 18))  # Make figure larger with width=15, height=10
    if len(entity_types) > 0:
        # Unzip the tuples into separate lists for x and y values
        class_names, counts = zip(*correct)
        incorrect_names, incorrect_counts = zip(*incorrect)
        for class_name in class_names:
            if class_name not in incorrect_names:
                incorrect_names = incorrect_names + (class_name,)
                incorrect_counts = incorrect_counts + (0,)
        for class_name in incorrect_names:
            if class_name not in class_names:
                class_names = class_names + (class_name,)
                counts = counts + (0,)
        # Create x positions for the bars
        x = np.arange(len(class_names))
        width = 0.35  # Width of the bars

        # Create the stacked bars
        plt.bar(x, counts, width, label='Correct')
        plt.bar(x, incorrect_counts, width, bottom=counts, label='Incorrect')

        # Customize the plot
        plt.xticks(x, class_names, rotation=90)
        plt.title(f"GRU_{i} Class Accuracy")
        plt.xlabel("Class Type")
        plt.ylabel("Count")
        plt.legend()
        plt.xticks(rotation=90)
        plt.savefig(f"GRU_{i}_class_accuracy.png")
        plt.close()

    accuracy_results_gru.append(f"GRU_{i} accuracy: {ner_accuracy(pred_entities, true_entities)}")
    class_accuracy_results_gru.append(f"GRU_{i} class accuracy: {class_acc}")

for acc in zip(accuracy_results_gru):
    print(acc)

#for acc in zip(class_accuracy_results_gru):
    #print(acc)

#_ , gru_ner_data = ner_accuracy(pred_entities, true_entities)

#for acc in zip(gru_ner_data):
    #print(acc)


"""

#Uncomment this block to see translate_entities in action

example = [[('North America', 'Thing'), ('John', 'Person')],[('North America', 'Thing')]]

ents = translate_entities(example)
print(ents) 

print(ner_baseline('en_coNER_train.json', 20))
"""