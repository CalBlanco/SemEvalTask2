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
        for entity in sentence:
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