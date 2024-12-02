from rapidfuzz import fuzz

#test_predictions = [[("robert gottschalk", "OtherPER"), ("mila kunis", "OtherPER")],[("quinton tarantino", "Person")]]
#test_targets = [[("robert gottschalk", "OtherPER"), ("mila kunis", "OtherPER")],[("Michael Bay", "Person")]]

def ner_accuracy(predictions, targets) -> float:
    """
    Calculate the accuracy of the predictions compared to the targets

    **ARGS**
    predictions: list of a list of tuples with the prediction and the type of entity
    targets: list of a list of tuples with the target and the type of entity
    
    **RETURNS**
    accuracy: float of the accuracy of the predictions
    """
    correct = 0
    incorrect = 0
    sentence_count = 0
    for sentence in predictions:
        entity_count = 0
        if sentence == []:
            if targets[sentence_count] == []:
                correct += 1
            else:
                incorrect += 1
            sentence_count += 1
            continue
        for entity in sentence:
            if targets[sentence_count] == []:
                incorrect += 1
            elif entity_count >= len(targets[sentence_count]):
                for entity_target in targets[sentence_count]:
                    if fuzz.ratio(entity[0], entity_target[0]) > 80:
                        correct += 1
                    else:
                        incorrect += 1
            else:
                print(entity, targets[sentence_count][entity_count])
                if fuzz.ratio(entity[0], targets[sentence_count][entity_count][0]) > 80:
                    correct += 1
                else:
                    incorrect += 1
            entity_count += 1
        sentence_count += 1
    print(f"Correct: {correct}, Incorrect: {incorrect}")
    accuracy = correct/(correct+incorrect)
    print(f"Accuracy: {accuracy}")
    return accuracy

# example usage
# print(ner_accuracy(test_predictions, test_targets))

def class_accuracy(predictions, targets) -> float:
    """
    Calculate the accuracy of the predictions compared to the targets

    **ARGS**
    predictions: list of a list of tuples with the prediction and the type of entity
    targets: list of a list of tuples with the target and the type of entity

    **RETURNS**
    accuracy: float of the accuracy of the predictions
    entity_types: dictionary with the type of entity and the number of correct and incorrect predictions
    """
    correct = 0
    incorrect = 0
    sentence_count = 0
    entity_types = {}
    for sentence in predictions:
        entity_count = 0
        if sentence == []:
            if targets[sentence_count] == []:
                correct += 1
            else:
                incorrect += 1
            sentence_count += 1
            continue
        for entity in sentence:
            if targets[sentence_count] == []:
                entity_types[f"None_incorrect"] = entity_types.get(f"None_incorrect", 0) + 1
                incorrect += 1
            elif entity_count >= len(targets[sentence_count]):
                for entity_target in targets[sentence_count]:
                    if fuzz.ratio(entity[1], entity_target[1]) > 80:
                        entity_types[f"{entity[1]}_correct"] = entity_types.get(f"{entity[1]}_correct", 0) + 1
                        correct += 1
                    else:
                        if entity[1] == "":
                            entity_types[f"None_incorrect"] = entity_types.get(f"None_incorrect", 0) + 1
                        else:
                            entity_types[f"{entity[1]}_incorrect"] = entity_types.get(f"{entity[1]}_incorrect", 0) + 1
                        incorrect += 1
            else:
                print(entity, targets[sentence_count][entity_count])
                if fuzz.ratio(entity[1], targets[sentence_count][entity_count][1]) > 80:
                    entity_types[f"{entity[1]}_correct"] = entity_types.get(f"{entity[1]}_correct", 0) + 1
                    correct += 1
                else:
                    if entity[1] == "":
                        entity_types[f"None_incorrect"] = entity_types.get(f"None_incorrect", 0) + 1
                    else:
                        entity_types[f"{entity[1]}_incorrect"] = entity_types.get(f"{entity[1]}_incorrect", 0) + 1
                    incorrect += 1
            entity_count += 1
        sentence_count += 1
    print(f"Correct: {correct}, Incorrect: {incorrect}")
    accuracy = correct/(correct+incorrect)
    print(f"Accuracy: {accuracy}")
    return accuracy, entity_types
