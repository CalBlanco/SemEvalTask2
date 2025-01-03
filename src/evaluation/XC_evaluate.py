#
# This script is based on the evaluation script provided by Apple for the SemEval 2025 Task 2.
# The original script can be found at (https://github.com/apple/ml-kg-mt/src/evaluation/evaluate.py).   
#

# Predictions Format
# {
#   "id": "Q627784_0",
#   "prediction": "Come viene ricordato e onorato Yu il Grande nella storia e cultura cinese di oggi?",
# }


"""
evaluate.py

This script evaluates the predictions of the model against the references of the test set of XC-Translate.

The script outputs manual entity translation accuracy (m-ETA). The m-ETA is computed by comparing the predicted entity names with the ground truth entity names, which have been manually-curated by human annotators. At a high-level, m-ETA provides an indication of quality of the translation of an entity name. As such, it does not provide a translation-level score and, therefore, it is recommended to use m-ETA together with other metrics. The script also allows filtering the evaluation to a specific entity type.

Usage:
    python evaluate.py --references <references> --predictions <predictions> [--entity_types <entity_types>] [--verbose]

Arguments:
    --references (str): Path to the references.
    --predictions (str): Path to the predictions.
    --entity_types (str, optional): Evaluate only on the specified entity type(s). Choices: "Musical work", "Artwork", "Food", "Animal", "Plant", "Book", "Book series", "Fictional entity", "Landmark", "Movie", "Place of worship", "Natural place", "TV series", "Person".
    --verbose: Set this flag to print every wrong match.

Minimal example:
    python src/evaluation/evaluate.py --references data/xct/references/all/it_IT.jsonl --predictions data/xct/predictions/to_it_IT/it_IT.gpt-4.json

Full example:
    python src/evaluation/evaluate.py --references data/xct/references/all/it_IT.jsonl --predictions data/xct/predictions/to_it_IT/it_IT.gpt-4.json --entity_types "Movie" "TV series" --verbose
"""

import argparse
import json
import re
import sys
from typing import Dict, List, Set
from nltk.translate.bleu_score import sentence_bleu

from loguru import logger

fmt = "<level>{level}</level> | {message}"
logger.remove()
logger.add(sys.stderr, format=fmt)


# List of entity types in the XC-Translate dataset.
# Used to filter the evaluation to a specific entity type.
ENTITY_TYPES = [
    "Musical work",
    "Artwork",
    "Food",
    "Animal",
    "Plant",
    "Book",
    "Book series",
    "Fictional entity",
    "Landmark",
    "Movie",
    "Place of worship",
    "Natural place",
    "TV series",
    "Person",
]


def load_references(input_path: str, entity_types: List[str]) -> List[dict]:
    """
    Load data from the input file (JSONL) and return a list of dictionaries, one for each instance in the dataset.

    Args:
        input_path (str): Path to the input file.
        entity_types (List[str]): List of entity types to filter the evaluation.

    Returns:
        List[dict]: List of dictionaries, one for each instance in the dataset.
    """
    data = []

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line_data = json.loads(line)

            # Skip instances with empty target list and log a warning.
            if not line_data["targets"]:
                logger.warning(f"Empty target list for instance {line_data['id']}")
                continue

            # Filter the evaluation to the specified entity types if provided.
            if entity_types and not any(
                e in line_data["entity_types"] for e in entity_types
            ):
                continue

            data.append(line_data)

    return data


def load_predictions(input_path: str) -> Dict[str, str]:
    """
    Load data from the input file (JSONL) and return a dictionary with the instance ID as key and the prediction as value.

    Args:
        input_path (str): Path to the input file.

    Returns:
        Dict[str, str]: Dictionary with the instance ID as key and the prediction as value.
    """
    data = {}

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line_data = json.loads(line)
            prediction = line_data["prediction"]

            # Get the instance ID from a substring of the ID.
            pattern = re.compile(r"Q[0-9]+_[0-9]")
            match = pattern.match(line_data["id"])
            if not match:
                raise ValueError(f"Invalid instance ID: {line_data['id']}")

            instance_id = match.group(0)
            data[instance_id] = prediction

    return data


def compute_entity_name_translation_accuracy(
    predictions: Dict[str, str],
    mentions: Dict[str, Set[str]],
    verbose: bool = False,
    comparison_type: str = "mta",
) -> dict:
    """
    Compute the entity name translation accuracy.

    Args:
        predictions (Dict[str, str]): Predictions of the model.
        mentions (Dict[str, Set[str]]): Ground truth entity mentions.
        verbose (bool): Set to True to print every wrong match.

    Returns:
        dict: Dictionary with the following
            - correct: Number of correct matches.
            - total: Total number of instances.
            - accuracy: Accuracy of the model.
    """
    if comparison_type == "mta":
        correct, total = 0, 0

        for instance_id, instance_mentions in mentions.items():
            # Check that there is at least one entity mention for the instance.
            assert instance_mentions, f"No mentions for instance {instance_id}"

            # Increment the total count of instances (for recall calculation).
            total += 1

            # Check that there is a prediction for the instance.
            if instance_id not in predictions:
                if verbose:
                    logger.warning(
                        f"No prediction for instance {instance_id}. Check that this is expected behavior, as it may affect the evaluation."
                    )
                continue

            prediction = predictions[instance_id]
            normalized_translation = prediction.casefold()
            entity_match = False

            for mention in instance_mentions:
                normalized_mention = mention.casefold()

                # Check if the normalized mention is a substring of the normalized translation.
                # If it is, consider the prediction (the entity name translation) correct.
                if normalized_mention in normalized_translation:
                    correct += 1
                    entity_match = True
                    break

            # Log the prediction and the ground truth mentions for every wrong match if verbose is set.
            if not entity_match and verbose:
                logger.info(f"Prediction: {prediction}")
                logger.info(f"Ground truth mentions: {instance_mentions}")
                logger.info("")

        return {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total > 0 else 0.0,
        }
    elif comparison_type == "bleu":
        cumulative_bleu_score = 0.0
        total = 0
        for instance_id, instance_sentence in mentions.items():
            # Check that there is a prediction for the instance.
            if instance_id not in predictions:
                if verbose:
                    logger.warning(
                        f"No prediction for instance {instance_id}. Check that this is expected behavior, as it may affect the evaluation."
                    )
                continue

            prediction = predictions[instance_id]
            normalized_translation = prediction.casefold()
            bleu_score = sentence_bleu(instance_sentence, normalized_translation)
            total += 1
            cumulative_bleu_score += bleu_score
        
        if total == 0:
            return {
                "average_bleu_score": 0.0,
            }
        else:
            return {
                "average_bleu_score": cumulative_bleu_score / total,
        }


def get_mentions_from_references(data: List[dict]) -> Dict[str, Set[str]]:
    """
    Load the ground truth entity mentions from the data.

    Args:
        data (List[dict]): List of dictionaries, one for each instance in the dataset.

    Returns:
        Dict[str, Set[str]]: Dictionary with the instance ID as key and the set of entity mentions as value.
    """
    mentions = {}

    for instance in data:
        instance_id = instance["id"]
        instance_mentions = set()

        for target in instance["targets"]:
            mention = target["mention"]
            instance_mentions.add(mention)

        mentions[instance_id] = instance_mentions

    return mentions

def get_sentence_from_references(data: List[dict]) -> Dict[str, Set[str]]:
    """
    Load the ground truth sentence from the data.

    Args:
        data (List[dict]): List of dictionaries, one for each instance in the dataset.

    Returns:
        Dict[str, Set[str]]: Dictionary with the instance ID as key and the set of entity sentences as value.
    """
    sentences = {}

    for instance in data:
        instance_id = instance["id"]
        instance_sentences = set()

        for target in instance["targets"]:
            sentence = target["translation"]
            instance_sentences.add(sentence)

        sentences[instance_id] = instance_sentences

    return sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--references",
        type=str,
        required=True,
        help="Path to the references.",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to the predictions.",
    )
    parser.add_argument(
        "--entity_types",
        type=str,
        nargs="*",
        required=False,
        default=None,
        choices=ENTITY_TYPES,
        help="Evaluate only on the specified entity type(s).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set this flag to print every wrong match.",
    )

    args = parser.parse_args()

    logger.info("Running evaluation on XC-Translate with the following parameters:")
    logger.info(f" - References: {args.references}")
    logger.info(f" - Predictions: {args.predictions}")
    logger.info(f" - Entity types: {args.entity_types}")
    logger.info(f" - Verbose: {args.verbose}")
    logger.info("")

    logger.info(f"Loading data from {args.references}...")
    reference_data = load_references(args.references, args.entity_types)
    mentions = get_mentions_from_references(reference_data)
    sentences = get_sentence_from_references(reference_data)
    assert len(mentions) == len(reference_data)
    logger.info(f"Loaded {len(reference_data)} instances.")

    logger.info(f"Loading data from {args.predictions}...")
    prediction_data = load_predictions(args.predictions)
    logger.info(f"Loaded {len(prediction_data)} predictions.")

    logger.info("Computing entity name translation accuracy...")

    # you can swap the "sentences" and "mentions" arguments as well as "mta" and "bleu" arguments to evaluate whole sentences or
    # just mentions as well as using mta or bleu to compare accuracy
    entity_name_translation_accuracy = compute_entity_name_translation_accuracy(
        prediction_data,
        sentences,
        args.verbose,
        comparison_type="bleu",
    )

    logger.info("")
    logger.info("=============================================")
    logger.info("Evaluation results:")
    logger.info(f"Average BLEU score = {entity_name_translation_accuracy['average_bleu_score']:.2f}")
    """
    logger.info(f"Correct instances   = {entity_name_translation_accuracy['correct']}")
    logger.info(f"Total instances     = {entity_name_translation_accuracy['total']}")

    accuracy = entity_name_translation_accuracy["accuracy"] * 100.0
    logger.info("-----------------------------")
    logger.info(f"m-ETA               = {accuracy:.2f}")
    logger.info("=============================================")
    logger.info("")
"""
    logger.info("Evaluation completed.")