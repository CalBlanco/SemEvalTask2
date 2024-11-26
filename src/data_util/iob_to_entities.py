import json
import os
import pandas as pd

def iob_to_entities(dataframe)->list[list[tuple[str, str]]]:
    """Convert IOB format to entities
    
    ARGS:
        dataframe -- pandas dataframe or list of tuples with sentences and IOB formated tags
    """
    sources = []
    targets = []
    if isinstance(dataframe, pd.DataFrame):
        for id, row in dataframe.iterrows():
            if id == "ID":
                continue
            else:
                sources.append(row['Sentence'])
                targets.append(row['IOB Slot tags'])
    elif isinstance(dataframe, list):
        for datapoint in dataframe:
            sources.append(datapoint[0])
            targets.append(datapoint[1])
    else:
        raise ValueError("dataframe must be a pandas dataframe or list of tuples")
    
    data = list(zip(sources, targets))
    
    data_entities = []
    for source, tags in data:
        entities_found = []
        named_entity = ""
        for i in range(len(tags)):
            tag = tags[i]
            if tag[0] == 'B':
                named_entity += source[i] + " "
                tag_type = tag[2:]
            elif tag[0] == 'I':
                named_entity += source[i] + " "
            elif tag[0] == 'O' and named_entity != "":
                entities_found.append((named_entity[:-1], tag_type))
                named_entity = ""
        if named_entity != "":
            entities_found.append((named_entity[:-1], tag_type))

        data_entities.append(entities_found)

    return data_entities

# example usage
#test = [(["I", "really", "like", "the", "Hangover"], ["O", "O", "O", "B_Movie", "I_Movie"]), (["Quinton", "Tarantino", "is", "a", "great", "director"], ["B_Person", "I_Person", "O", "O", "O", "O"])]
#testdataframe = pd.DataFrame(test, columns=["Sentence", "IOB Slot tags"])
#print(iob_to_entities(test))