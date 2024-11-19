from datasets import load_dataset, load_dataset_builder
import json

en_dataset = load_dataset("multiCoNER/multiconer_v2", "English (EN)", trust_remote_code=True)

def save_to_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)


train_data = [{"id": instance["id"], "sample_id": instance["sample_id"], "tokens": instance["tokens"], "ner_tags": instance["ner_tags"], "ner_tags_index": instance["ner_tags_index"]} for instance in en_dataset["train"]]
validation_data = [{"id": instance["id"], "sample_id": instance["sample_id"], "tokens": instance["tokens"], "ner_tags": instance["ner_tags"], "ner_tags_index": instance["ner_tags_index"]} for instance in en_dataset["validation"]]

test_data = [
    {"id": instance["id"], "sample_id": instance["sample_id"], "tokens": instance["tokens"], "ner_tags": instance["ner_tags"], "ner_tags_index": instance["ner_tags_index"]}
    for i, instance in enumerate(en_dataset["test"]) if i < 100000
]

# save_to_json(train_data, "en_coNER_train.json")
# save_to_json(validation_data, "en_coNER_validation.json")
save_to_json(test_data, "en_coNER_test.json")