import json

def retrieve_coNER(file_path: str):
    """
    Retrieves tokens and target IOB ner_tags from coNER file
    """

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        tokens = [item['tokens'] for item in data]
        ner_tags = [item['ner_tags'] for item in data]

        return tokens, ner_tags
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return [], []
    except KeyError as e:
        print(f"Error: Missing key in JSON data - {e}")
        return [], []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return [], []

# Example usage
file_path = 'coNER_data/en_coNER_train.json'
tokens, ner_tags = retrieve_coNER(file_path)

print(f"Tokens: {tokens[:2]}")  # Print first 2 samples
print(f"NER Tags: {ner_tags[:2]}")