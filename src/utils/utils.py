##!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Miscellaneous utilities for model training, evaluation and storing.

@author lisa.raithel@dfki.de
"""

import csv
import json
import numpy as np
import os
import sys


def take_over_config_details(config, results_file):
    """Take over as many details as possible from the config file.

    Args:
        config: dict
                The input configuration
        results_file:   str
                        The file were the all results are written to.

    Returns:
        A dictionary containing all values as given in the configuration.
    """
    results_dict = {}

    with open(results_file, "r") as csv_file:
        header = csv.DictReader(csv_file).fieldnames

    for key in header:
        if key in config:
            results_dict[key] = config[key]
    return results_dict


def save_results(results_file, results_dict):
    """Save all results to the same CSV.

    Args:
        results_file:   str
                        The CSV for the results.
        results_dict:   dict
                        A dict with all results.
    """
    header = [
        "model_name", "pretrained", "date", "time", "classif_threshold",
        "determined_threshold", "cross_val", "early_stopping", "epochs",
        "mcg_file_header", "min_delta", "min_num_inst", "min_rep",
        "num_hidden_layers", "max_length_of_instances", "patience",
        "regularizers", "train_set_size", "num_concepts_overall",
        "num_instances_overall", "weighted_f1_old_ts_best_model",
        "weighted_f1_old_ts_average", "precision_weighted", "precision_macro",
        "precision_micro", "recall_weighted", "recall_macro", "recall_micro",
        "f1_weighted", "f1_macro", "f1_micro",
        "num_instances_not_in_embedding_vocab", "batch_size",
        "use_class_weights", "embedding_model", "embedding_combiner",
        "check_other_embedding", "min_count_of_instances",
        "avrg_concepts_per_inst", "avrg_inst_per_concept", "learning_rate",
        "num_train_val_instances", "num_test_instances"
    ]

    for key in header:
        if key not in results_dict:
            assert False, "key {} missing".format(key)

    # write results dict to results file
    with open(results_file, 'a', newline='') as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=header)

        writer.writerow(results_dict)


def read_config(config_file):
    """Return config JSON as dict.

    Args:
        config_file:    str
                        The path to a JSON file.
    Returns:
        The configuration as a dict.
    """
    print("config file: {}".format(config_file))
    with open(config_file, "r") as read_handle:
        config = json.load(read_handle)

    return config


def save_config_to_file(config, model_path, details_str):
    """Save the configuration in the model folder.

    Args:
        config: dict
                The configuration of the current model.
        model_path: str
                    The path to the directory of the model.
    """
    config_file_name = "{}/config_{}.json".format(model_path, details_str)

    with open(config_file_name, "w") as write_handle:
        json.dump(config, write_handle)


def create_dir(min_rep, min_num_instances, num_hidden, regularizers, timestr):
    """Create a directory with the name of all given parameters."""
    l2_regs_str = "-".join([str(x) for x in regularizers])

    model_path = "results/minrep{}_mininst{}_hidden{}_reg{}_{}".format(
        min_rep, min_num_instances, num_hidden, l2_regs_str, timestr)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print("Created folder '{}'.".format(model_path))
        return model_path, "minrep{}_mininst{}_hidden{}_reg{}_{}".format(
            min_rep, min_num_instances, num_hidden, l2_regs_str, timestr)

    print("Folder '{}' does already exist.".format(model_path))
    sys.exit(1)


def write_instances_to_file(indices,
                            rep_matrix,
                            model_path,
                            details_str,
                            id2word_dict,
                            concepts,
                            mode="test"):
    """Write all test words and their associated concepts to a tsv file.

    Args:
        indices:    list
                    A list of test instance indices.
        rep_matrix: numpy ndarray
                    A matrix of multi-hot encoded concepts-
        model_path: str
                    The path to the model.
        id2word_dict:   dict
                        A dictionary mapping IDs to instances.
        concepts:   list
                    A list of all concepts (labels).
    """
    instances = []
    with open("{}/{}_set_{}.csv".format(model_path, mode, details_str),
              "w") as write_handle:

        for idx in indices:
            instance = id2word_dict[idx]
            print(instance)
            instances.append(instance)
            reps = np.asarray(rep_matrix[idx])[0]
            print(reps)
            # concept row
            concepts_per_instance = [x for x, z in zip(concepts, reps) if z]
            write_handle.write("{}\t{}\n".format(
                instance, "\t".join(concepts_per_instance)))

    return instances


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncate a sequence pair in place to the maximum length.

    Taken from https://github.com/huggingface/pytorch-pretrained-BERT/blob
    /master/examples/extract_features.py

    This is a simple heuristic which will always truncate the longer sequence
    one token at a time. This makes more sense than truncating an equal
    percent of tokens from each, since if one sequence is very short then each
    token that's truncated likely contains more information than a longer
    sequence.
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def sanity_checks(x_train, y_train, x_dev, y_dev, x_test, y_test, embed_matrix,
                  rep_matrix, concepts, idx_train, idx_dev, idx_test,
                  id2word_dict):
    """Check all data for overlaps etc."""
    print("Sanity checks.\n")
    print("x_train shape: {}, y_train shape: {}".format(
        x_train.shape, y_train.shape))
    print(("x_dev shape: {}, y_dev shape: {}\n"
           "x_test shape: {}, y_test shape: {}").format(
               x_dev.shape, y_dev.shape, x_test.shape, y_test.shape))

    assert x_dev.shape[0] >= 1 and x_test.shape[0] >= 1, (
        "Please change the training data size, there is not enough data "
        "for the dev and test sets.")
    assert len(x_train) + len(x_dev) + len(x_test) == len(embed_matrix)
    assert len(y_train) + len(y_dev) + len(y_test) == len(rep_matrix)
    assert y_train.shape[1] == len(concepts)

    train_instances = set()
    for idx_tr in idx_train:
        train_instances.add(id2word_dict[idx_tr])

    dev_instances = set()
    for idx_dev in idx_dev:
        dev_instances.add(id2word_dict[idx_dev])

    test_instances = set()
    for idx_te in idx_test:
        test_instances.add(id2word_dict[idx_te])

    # make sure the sets are disjoint
    assert not train_instances.intersection(
        dev_instances
    ), "Train and Dev data are overlapping, please check the data split."
    assert not train_instances.intersection(
        test_instances
    ), "Train and Test data are overlapping, please check the data split."
    assert not dev_instances.intersection(
        test_instances
    ), "Dev and Test data are overlapping, please check the data split."
