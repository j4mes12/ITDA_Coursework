import numpy as np
import re

# Use HuggingFace's datasets library to access the financial_phrasebank dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def import_task1_data():
    """The financial_phrasebank dataset is available in four variations.
    It has no predefined train/validation/test splits. Each data point
    was annotated by 5-8 people, then their annotations were combined.
    Each variation of the dataset contains examples with different levels
    of agreement. Let's use the one containing all data points where at
    least 50% of the annotators agreed on the label."""

    dataset = load_dataset(
        "financial_phrasebank",
        "sentences_50agree",  # Select variation of the dataset
    )

    print(f"The dataset is a dictionary with two splits: \n\n{dataset}")

    # Split test data from training data
    (
        train_sentences,
        test_sentences,
        train_labels,
        test_labels,
    ) = train_test_split(
        dataset["train"]["sentence"],
        dataset["train"]["label"],
        test_size=0.2,
        stratify=dataset["train"]["label"],
    )

    # label 0 = negative, 1 = neutral, 2 = positive
    print(
        f"How many instances in the train dataset? \n\n{len(train_sentences)}"
    )
    print("")
    print(f"What does one instance look like? \n\n{train_sentences[234]}")

    (
        train_sentences,
        val_sentences,
        train_labels,
        val_labels,
    ) = train_test_split(
        train_sentences, train_labels, test_size=0.25, stratify=train_labels
    )

    print(f"instances in the validation dataset \n\n{len(val_sentences)}\n")
    print(f"instances in the test dataset \n\n{len(test_sentences)}")

    return (
        train_sentences,
        test_sentences,
        val_sentences,
        train_labels,
        test_labels,
        val_labels,
    )


def read_sec_filings(split):
    # Use this function to load the SEC filings data from text files

    if split == "train":
        with open("./data/SEC-filings/train/FIN5.txt") as fp:
            lines = fp.readlines()
    else:
        with open("./data/SEC-filings/test/FIN3.txt") as fp:
            lines = fp.readlines()

    # store the tokens and labels for all sentences
    sentences = []
    labels = []

    # the tokens and labels for the current sentence
    current_sen = []
    current_labels = []

    for i in range(2, len(lines)):
        # print(f'This is line {i}')
        # print(lines[i])

        if (
            len(lines[i]) > 1
        ):  # Line with some data on: The data consists of tokens and tags.
            data = re.split(" ", lines[i])  # tokenise the line
            # print(data)
            current_sen.append(data[0])  # append the token

            # data[1] contains POS tags. you can also use these in your model.

            current_labels.append(data[3].strip())  # append the NER tag
        elif len(current_sen) > 1:  # this marks the end of a sentence
            # end of sentence
            sentences.append(current_sen)  # save the tokens for this sentence
            current_sen = []  # reset

            labels.append(current_labels)  # save the tags for this sentence
            current_labels = []

    if len(current_sen) > 1:  # save the last sentence
        sentences.append(current_sen)
        labels.append(current_labels)

    print(f"Number of sentences loaded = {len(sentences)}")
    print(f"Number of unique labels: {np.unique(np.concatenate(labels))}")

    return sentences, labels


def import_task2_data():
    print("Loading the original training set: ")
    sentences_ner, labels_ner = read_sec_filings("train")

    print("\nLoading the test set: ")
    test_sentences_ner, test_labels_ner = read_sec_filings("test")

    (
        train_sentences_ner,
        val_sentences_ner,
        train_labels_ner,
        val_labels_ner,
    ) = train_test_split(
        sentences_ner,
        labels_ner,
        test_size=0.2,
    )

    return (
        train_sentences_ner,
        train_labels_ner,
        test_sentences_ner,
        test_labels_ner,
        val_sentences_ner,
        val_labels_ner,
    )
