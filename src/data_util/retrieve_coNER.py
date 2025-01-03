import json
import os


path = os.path.join(os.path.dirname(__file__), '../../data/coNER_data/')

def retrieve_coNER(file: str):
    """
    Retrieves tokens and target IOB ner_tags from coNER file
    """

    try:
        file_path = os.path.join(path, file)
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
# tokens, ner_tags = retrieve_coNER("en_coNER_train.json")

# print(f"Tokens: {tokens[:2]}")  # Print first 2 samples
# print(f"NER Tags: {ner_tags[:2]}")