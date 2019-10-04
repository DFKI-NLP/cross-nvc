##!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A test script for experiments with NVC.

@author lisa.raithel@dfki.de
"""
import argparse

from models.nvc_model import NeuralVectorConceptualizer
from models.embeddings import Embedding

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data", type=str, help="The path to the tsv/json data.")

    parser.add_argument(
        "-embedding_or_raw_data_file",
        type=str,
        default="",
        help="The path to the embedding data [fasttext].")

    parser.add_argument("config", type=str, help="The configuration file.")

    parser.add_argument(
        "model_name",
        type=str,
        default="fasttext",
        help=("The name of the embedding model to be used [fasttext or "
              "word2vec or bert]."))

    ARGS, _ = parser.parse_known_args()

    embedding = Embedding(
        embedding_or_raw_data_file=ARGS.embedding_or_raw_data_file,
        voc_limit=100000,
        model_name=ARGS.model_name,
        config_file=ARGS.config)

    nvc = NeuralVectorConceptualizer(
        config_file=ARGS.config, embedding=embedding, debug=False)

    # load the unfiltered data (either as TSV or as JSON)
    # filter data according to criteria in config file
    # (min_rep & min_instances)
    nvc.load_data(path_to_data=ARGS.data, filtered=False)
    # load the already filtered data
    # nvc.load_data(path_to_data=ARGS.data, filtered=True)

    # compare predictions and ground truth manually with 'inspect_concept'
    nvc.train()
    # specify the paths to the pre-trained model and the test data
    # model_path = ""

    # nvc.load_pretrained_model(trained_model=model_path, x_val_file="")

    # data = ""
    # nvc.load_pretrained_model(trained_model=model_path, x_val_file=data)
    nvc.show_activations(["stone", "sun"], max_highlights=3, plot_cosine=True)
