#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A simple classifier for neural word vector conceptualization (NVC).

@author lisa.raithel@dfki.de
"""

import argparse
import json
import numpy as np
import logging
import pandas as pd
import time
import sys

from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from keras import regularizers, optimizers
from keras import backend as keras_backend
from tensorflow.python.client import device_lib

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight

from sklearn.model_selection import KFold
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             precision_recall_fscore_support)

# own modules
from data.prepare_corpus import DataLoader
from utils.callbacks_and_metrics import (MetricsCallback, compute_roc_auc,
                                         determine_best_threshold,
                                         plot_precision_recall_curve)
from utils import utils
from utils.visualizations import Visualizer

try:
    from livelossplot.keras import PlotLossesCallback
    live_plotting_available = True
except ImportError:
    live_plotting_available = False

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuralVectorConceptualizer(object):
    """A model for neural vector conceptualization of word vectors."""

    def __init__(self, config_file, embedding, debug=False):
        """Initialize the NVC model class.

        Args:
            config_file:    str
                            The path to a configuration in JSON format.
            embedding:      class
                            A class providing embedding utilities.
        """
        print("local devices: {}".format(device_lib.list_local_devices()))
        keras_backend.tensorflow_backend._get_available_gpus()

        self.debug = debug
        self.embedding = embedding
        self.word_vectors = embedding.get_word_vectors()
        self.embedding_dim = embedding.embedding_dim

        # get the configuration
        self.config = utils.read_config(config_file)
        self.timestr = time.strftime("%Y_%m_%d-%H_%M")
        self.threshold = self.config["classif_threshold"]
        self.pretrained = False

        self.model = None
        self.seed = 42
        self.model_path = ""
        self.model_name = ""
        self.details_str = ""
        self.results_file = "results/results_all_models.csv"
        self.results_to_write = utils.take_over_config_details(
            self.config, self.results_file)
        self.results_to_write["date"] = self.timestr.split("-")[0]
        self.results_to_write["time"] = self.timestr.split("-")[1]
        self.results_to_write["pretrained"] = self.pretrained
        self.results_to_write[
            "embedding_model"] = self.embedding.embedding_model_name
        self.results_to_write[
            "embedding_combiner"] = self.embedding.embedding_combiner

        self.word2id = {}
        self.id2embedding = {}
        self.id2word = {}

        self.visualizer = Visualizer(
            model_path=self.model_path,
            details_str=self.details_str,
            debug=debug)

    def create_model_name_and_dir(self):
        """Create a directory for new trained model and associated files."""
        if not self.debug:
            self.model_path, self.details_str = utils.create_dir(
                self.config["min_rep"], self.config["min_num_inst"],
                self.config["num_hidden_layers"], self.config["regularizers"],
                self.timestr)

            self.model_name = "{}/model_{}_{}.h5".format(
                self.model_path, self.details_str,
                self.embedding.embedding_model_name)

            self.results_to_write["model_name"] = "model_{}.h5".format(
                self.details_str)

            with open("{}/config.json".format(self.model_path),
                      'w') as json_file:
                json.dump(self.config, json_file)

            self.visualizer.update_paths(
                model_path=self.model_path,
                details_str=self.details_str,
                pretrained=self.pretrained)
        else:
            self.model_path = "debug_results"
            self.details_str = "debug"

    def load_data(self, path_to_data, filtered, selected_concepts=[]):
        """Load raw data via the DataLoader.

        Args:
            path_to_data:   string
                            The path to either a json or a tab-separated
                            file of the raw data with a header as given
                            in the configuration file.
            filtered:   bool
                        Is the given data already filtered or not.
        """
        # prepare the data loader
        self.loader = DataLoader(
            embedding=self.embedding,
            min_count_of_instances=self.config["min_count_of_instances"],
            use_logs=True,
            max_length_of_instances=self.config["max_length_of_instances"],
            check_concepts_for_cosine=self.config["check_concepts_for_cosine"],
            debug=self.debug)

        if filtered:
            self.filter_data(data=path_to_data)

        else:
            # load the raw data
            self.raw_data = self.loader.load_raw_data(
                path_to_data=path_to_data,
                header=self.config["mcg_file_header"],
                save_to_json=True,
                selected_concepts=selected_concepts)
            # and filter it according to the given criteria
            self.filter_data(data=self.raw_data)

    def filter_data(self, data={}):
        """Filter the dataset according to certain criteria.

        The minimum number of instances and the minimal REP values these
        instances have to have are extracted from the configuration file.
        Moreover, you can only train on a subset of concepts.
        Per default, all concepts that fulfill the configuration criteria are
        used for creating the dataset.

        Args:
            selected_concepts:  list, optional
                                Train only on a subset of concepts and their
                                respective instances.
        """
        # the minimum number of instances a concept needs to have to be
        # considered
        self.min_num_instances = self.config["min_num_inst"]
        # the minimal REP value the instances need to have to be 'accepted'
        self.min_rep = self.config["min_rep"]
        # get filtered data as pandas data frame:
        # rows: instances
        # columns: concepts
        (self.filtered_data_matrix, self.instances,
         self.concepts) = self.loader.load_filtered_data_matrix(
             data=data,
             min_num_instances=self.min_num_instances,
             min_rep=self.min_rep,
             save_to_json=True)

        print("shape filtered_data_matrix: {}".format(
            self.filtered_data_matrix.shape))
        print("#instances: {}".format(len(self.instances)))
        print("#concepts: {}".format(len(self.concepts)))

        file_name = "results/words_not_in_mcg_i{}_v{}_{}_{}.txt".format(
            self.min_num_instances, self.min_rep, self.timestr,
            self.embedding.embedding_model_name)

        # needs to be done only once per filtered data
        num_instances_not_in_w2v = self.loader.get_all_words_not_in_mcg(
            self.instances, file_name)

        self.results_to_write[
            "num_instances_not_in_embedding_vocab"] = num_instances_not_in_w2v

        # collect all instances and labels in lists and create lookup tables
        self.inst2id, self.id2inst = self.prepare_instances()
        self.concept2id, self.id2concept = self.prepare_concepts()

        self.results_to_write["num_concepts_overall"] = len(self.concepts)
        self.results_to_write["num_instances_overall"] = len(self.instances)

    def prepare_instances(self):
        """Create instance list, instance2id and id2instance dictionaries.

        Returns:
            instances:  list
                        A list of all instances.
            instance2id:    dict
                            A dictionary from instances to IDs.
            id2instance:    dict
                            A dictionary from IDs to instances.
        """
        # instances = list(self.filtered_data.index)
        instance2id = {}
        id2instance = {}

        for i, instance in enumerate(self.instances):
            instance2id[instance] = i
            id2instance[i] = instance

        return instance2id, id2instance

    def prepare_concepts(self):
        """Create concept list, concept2id and vice versa.

        Returns:
            labels: list
                    A list of all concepts (labels).
            concept2id:   dict
                        A dictionary from concepts to IDs.
            id2concept:   dict
                        A dictionary from IDs to concepts.
        """
        # labels = list(self.filtered_data)
        concept2id = {}
        id2concept = {}

        for i, concept in enumerate(self.concepts):
            concept2id[concept] = i
            id2concept[i] = concept

        return concept2id, id2concept

    def split_data(self,
                   embedding_matrix,
                   rep_matrix,
                   train_size,
                   shuffle=True):
        """Split in training, development and test data.

        Args:
            embedding_matrix:   numpy ndarray
                                A (num_instances x embedding_size) matrix
                                containing all instances as word vectors.
            rep_matrix: numpy ndarray
                        A (num_instances x num_concepts) matrix containing
                        all multi-hot encoded label vectors for all instances.
            train_size: float
                        The percentage of training examples.
            shuffle:    bool
                        Shuffle the dataset or not.

        Returns:
            x_train:    numpy ndarray
                        A matrix of training instances.
            y_train:    numpy ndarray
                        A matrix of training labels.
            x_dev:  numpy ndarray
                    A matrix of validation instances.
            y_dev:  numpy ndarray
                    A matrix of validation labels.
            idx_dev:    list
                        A list of indices for the dev data.
            x_test: numpy ndarray
                    A matrix of test instance.
            y_test: numpy ndarray
                    A matrix of test labels.
            test_inst_list: list
                            A list of test words (not word vectors).
        """
        # Get the split-up training instances/labels and the remaining data,
        # as well as their indices.
        # The remaining data will be split into validation and test set.
        (x_train, x_remaining, y_train, y_remaining, idx_train,
         idx_remaining) = train_test_split(
             embedding_matrix,
             rep_matrix,
             np.arange(0, len(embedding_matrix)),
             train_size=train_size,
             shuffle=shuffle,
             random_state=self.seed)

        # Split the remaining data into validation and test set (50/50).
        # Also, get the respective indices.
        x_dev, x_test, y_dev, y_test, idx_dev, idx_test = train_test_split(
            x_remaining,
            y_remaining,
            idx_remaining,
            train_size=0.5,
            shuffle=shuffle,
            random_state=self.seed)

        print("Split data.\n\nWriting training/dev/test data to files.\n\n")
        # write the test instances and their respective concepts to a file
        test_inst_list = utils.write_instances_to_file(
            indices=idx_test,
            rep_matrix=rep_matrix.todense(),
            model_path=self.model_path,
            details_str=self.details_str,
            id2word_dict=self.id2word,
            concepts=self.concepts,
            mode="test")

        # write the train instances and their respective concepts to a file
        utils.write_instances_to_file(
            indices=idx_train,
            rep_matrix=rep_matrix.todense(),
            model_path=self.model_path,
            details_str=self.details_str,
            id2word_dict=self.id2word,
            concepts=self.concepts,
            mode="train")

        # write the train instances and their respective concepts to a file
        utils.write_instances_to_file(
            indices=idx_dev,
            rep_matrix=rep_matrix.todense(),
            model_path=self.model_path,
            details_str=self.details_str,
            id2word_dict=self.id2word,
            concepts=self.concepts,
            mode="dev")

        # check again for data overlaps / size of datasets etc.
        utils.sanity_checks(
            x_train=x_train,
            y_train=y_train.todense(),
            x_dev=x_dev,
            y_dev=y_dev.todense(),
            x_test=x_test,
            y_test=y_test.todense(),
            embed_matrix=embedding_matrix,
            rep_matrix=rep_matrix.todense(),
            concepts=self.concepts,
            idx_train=idx_train,
            idx_dev=idx_dev,
            idx_test=idx_test,
            id2word_dict=self.id2word)

        return (x_train, y_train.toarray(), x_dev, y_dev.toarray(), idx_dev,
                x_test, y_test.toarray(), test_inst_list)

    def create_embedding_and_label_matrix(self, embedding_dim):
        """Create the dataset for further processing.

        Args:
            embedding_dim:  int
                            Embedding dimension of the embedding model used.
        Returns:
            embedding_matrix:   numpy ndarray
                                A matrix of size
                                (num_instances x embedding_dim).
            concept_reps_matrix:    numpy ndarray
                                    A matrix of size
                                    (num_instances x num_labels).
        """
        # initialize embedding and label matrices
        embedding_matrix = np.zeros((len(self.instances), embedding_dim))
        # concept_reps_matrix = np.zeros((len(self.instances),
        #                                 len(self.concepts)))

        print("Creating embedding and labels matrices ...")
        for i, word in enumerate(self.instances):
            if i % 10000 == 0:
                print(
                    "{}/{} instances done".format(i, len(self.instances)),
                    end="\r",  # noqa: E901
                    flush=True)  # noqa: E901
            # get the word vector for every instance
            embedding_vec = self.embedding.get_embedding_for_word(word)

            if embedding_vec is not None:
                # add the word vector to the embedding matrix and save
                # its ID for later lookups
                embedding_matrix[i] = embedding_vec
                self.id2embedding[i] = embedding_vec
                self.word2id[word] = i
                self.id2word[i] = word

                # # get the rep values for the current instance
                # # reps = np.array(self.filtered_data.loc[word])
                # reps = self.filtered_data_matrix[i]

                # # convert the rep vector in a multi-hot encoded vector
                # # reps[reps != 0.0] = 1
                # # count the number of activated concepts per instance
                # num_positive_labels += sum(reps)
                # # not really necessary to create this matrix a second time
                # # (it's the same as filtered_data_matrix), but just in case
                # # the embedding vector doesn't exist for any reason
                # concept_reps_matrix[i] = reps

        # num_positive_labels += sum(reps)

        avrg_num_concepts_per_instance = np.count_nonzero(
            self.filtered_data_matrix.todense()) / len(self.instances)
        print("\nDone.\nAvrg #positive concepts per instance: {}".format(
            avrg_num_concepts_per_instance))

        self.results_to_write[
            "avrg_concepts_per_inst"] = avrg_num_concepts_per_instance

        num_instances_per_concept = np.count_nonzero(
            self.filtered_data_matrix.todense(), axis=0).flatten()
        avrg_instances_per_concept = num_instances_per_concept.sum() / len(
            self.concepts)
        print("Avrg #instances per concept: {}".format(
            avrg_instances_per_concept))

        self.results_to_write[
            "avrg_inst_per_concept"] = avrg_instances_per_concept

        return embedding_matrix, self.filtered_data_matrix

    def get_model(self, num_output_units):
        """Build and compile the model.

        Args:
            num_output_units:   int
                                Number of output units for the classifier.
                                Corresponds to the number of classes.
        Returns:
            The compiled model.
        """
        # retrieve the list of l2 regularization factors and
        # number of hidden layers from the config file
        l2_regs = self.config["regularizers"]
        num_hidden_layers = self.config["num_hidden_layers"]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ MODEL ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # instantiate the keras input tensor with batches of
        # 300-dimensional vectors
        input_data = Input(shape=(self.embedding_dim, ), dtype="float32")

        x = input_data

        if num_hidden_layers != 0:

            # add fully connected layers without regularizers
            if len(l2_regs) == 0:
                for i in range(num_hidden_layers):
                    x = Dense(
                        units=num_output_units,
                        # keras default: Glorot uniform
                        kernel_initializer='glorot_uniform',  # xavier?
                        # keras default: zeros
                        bias_initializer='zeros',
                        activation="relu")(x)

            # add fully connected layers with l2 regularizers in each layer
            else:
                for i, reg in zip(range(num_hidden_layers), l2_regs):
                    x = Dense(
                        units=num_output_units,
                        activation="relu",
                        kernel_regularizer=regularizers.l2(reg))(x)
        # add an output layer with a sigmoid activation for each output
        # neuron
        output = Dense(units=num_output_units, activation="sigmoid")(x)

        model = Model(inputs=[input_data], outputs=[output])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        optim = optimizers.Adam(lr=self.config["learning_rate"], decay=1e-6)
        # optim = optimizers.SGD(
        #     lr=self.config["learning_rate"],
        #     decay=1e-6,
        #     momentum=0.9,
        #     nesterov=True)

        # compile the model
        model.compile(
            # binary cross entropy to account for independent concepts
            loss="binary_crossentropy",
            optimizer=optim,
            # calculate mean accuracy rate
            # across all predictions
            metrics=["categorical_accuracy"])

        return model

    def inspect_predictions_for_class(self, inspect_concept, x, x_idx, y):
        """Inspect the predictions manually.

        Prints tab-separated strings of instance-predicted-true to the screen.

        Args:
            inspect_concept:  str
                            The class to be inspected.
            x:  numpy ndarray
                The input data for the model.
            x_idx:  list
                    The indices of the input data.
            y:  numpy ndarray
                The label matrix for the input data.
        """
        try:
            # get the ID for the inspected class
            idx = self.concepts.index(inspect_concept)
        except ValueError as e:
            print(
                ("\nThe word {}.\nPlease choose one of the available labels.\n"
                 "Exit.").format(e))
            sys.exit(1)

        # predict the labels for the given data
        y_pred = self.model.predict(x)
        y_pred[y_pred >= self.threshold] = 1
        y_pred[y_pred < self.threshold] = 0

        # collect all predictions and the respective ground truths
        results = {}
        for i, word_idx in enumerate(x_idx):
            word = self.id2word[word_idx]
            results[word] = {"predicted": y_pred[i][idx], "true": y[i][idx]}

        print("\nConcept to be inspected: {}\n".format(inspect_concept))
        df = pd.DataFrame.from_dict(results, orient="index")
        print(df)
        print("\n")

    def check_vectors(self, vec=[], word=""):
        """Check if the vectors are valid.

        Check if the given word vectors/instances are part of the dataset
        and if they are part of the word2vec vocabulary.

        Args:
            vec_1:  numpy ndarray
                    A word vector.
            word_1: str
                    The actual word represented by the word vector.
        Returns:
            True if all word is not in train/dev/test data
            but in word2vec vocabulary, otherwise False.
        """
        # check if the word/vector was used before
        if word in self.word2id:
            print("Word '{}' in train/dev/test data, please choose "
                  "another word.".format(word))
#             return False

        # check if the vector is part of the word2vec vocabulary
        if vec is None:
            print("Word '{}' cannot be found in the embedding vocabulary, "
                  "please choose another word.".format(word))
            return False

        return True

    def sample_vectors(self, word_1, word_2, n_steps=3, model_name=""):
        """Sample the vectors between two vectors with n steps.

        Args:
            word_1: str
                    The first word.
            word_2: str
                    The last word.
            n_steps:    int
                        The number of vectors to be sampled between word_1
                        and word_2.
            model_name: str
                        The path to the model directory.
        """
        # get the word vectors for the given instances
        vec_1 = self.embedding.get_embedding_for_word(word_1)
        vec_2 = self.embedding.get_embedding_for_word(word_2)

        # check if instance is in training or test data
        for word, vec in [(word_1, vec_1), (word_2, vec_2)]:
            if word in self.word2id:
                print(
                    "The word '{}' is part of the train/dev/test data.".format(
                        word))

        # create dataframe for distances of target word to all concepts
        # cosine_dataframe = self.embedding.get_sim_btw_vector_and_concepts(
        #     vec_1, word_1, self.concepts)

        # create the embedding matrix as input for the model
        embedding_matrix = np.zeros((n_steps + 2, self.embedding_dim))
        # create an embedding matrix to feed into the model
        embedding_matrix[0] = vec_1
        # collect all known words + placeholders for the 'step-vectors'
        ticks_text = [word_1]

        for i in range(1, n_steps + 1):
            # calculate the next vector
            new_vec = (i / (n_steps + 1)) * (vec_2 - vec_1) + vec_1
            # add it to the embedding matrix
            embedding_matrix[i] = new_vec
            vec_name = "{}".format(i)
            # add a dummy name for the ticks in the plot
            ticks_text.append(vec_name)

            # get the similarities for the next vector
            # next_df = self.embedding.get_sim_btw_vector_and_concepts(
            #     new_vec, vec_name, self.concepts)
            # # concatenate the new dataframe with the old
            # cosine_dataframe = pd.concat([cosine_dataframe, next_df])

        # add the last vector to the embedding matrix and the last word to the
        # ticks list
        embedding_matrix[n_steps + 1] = vec_2
        ticks_text.append(word_2)

        # next_df = self.embedding.get_sim_btw_vector_and_concepts(
        #     vec_2, word_2, self.concepts)
        # cosine_dataframe = pd.concat([cosine_dataframe, next_df])

        # TODO: this is redundant
        # x = np.expand_dims(embedding_matrix, axis=2)
        predictions = self.model.predict(embedding_matrix)
        # create a dataframe from all predictions
        predictions_dataframe = pd.DataFrame(
            predictions, columns=self.concepts, index=ticks_text)

        print(predictions_dataframe)

        # visualize the samples for NVC and cosine similarity
        self.visualizer.show_samples(
            activations=predictions_dataframe,
            # similarities=cosine_dataframe,
            concepts=self.concepts,
            ticks_text=ticks_text,
            first_word=word_1,
            last_word=word_2,
            steps_between=n_steps)

    def show_activations(self,
                         instances,
                         max_highlights=3,
                         plot_cosine=False,
                         language="english"):
        """Plot the activations of an instance.

        Args:
            instances:  list
                        A list of instances whose activations should be
                        plotted.
            max_highlights: int
                            The number of labeled instances.
        """
        embedding_matrix = np.zeros((len(instances), self.embedding_dim))

        instances_in_embedding = []
        # check if instance is in training or test data
        for i, word in enumerate(instances):
            vec = self.embedding.get_embedding_for_word(word)

            if self.check_vectors(word=word, vec=vec):
                embedding_matrix[i] = vec
                instances_in_embedding.append(word)

        # TODO: this is redundant
        # x = np.expand_dims(embedding_matrix, axis=2)
        # predict the concept activations
        predictions = self.model.predict(embedding_matrix)

        if plot_cosine:
            # for comparison purposes, also plot the most
            # cosine-similar concepts
            similarities = []

            for instance in instances_in_embedding:
                sim = self.embedding.get_similarity_to_concepts(
                    instance, self.concepts)
                similarities.append(sim)

            self.visualizer.display_concept_activations(
                instances=instances_in_embedding,
                classes=self.concepts,
                predictions=predictions,
                similarities=similarities,
                max_highlights=max_highlights)
        else:
            self.visualizer.display_concept_activations_without_cosine(
                instances=instances_in_embedding,
                classes=self.concepts,
                predictions=predictions,
                max_highlights=max_highlights,
                language=language)

    def prepare_vectors(self, vectors_as_strings):
        """Convert a vector in string format into an actual vector."""
        x_val = np.zeros((len(vectors_as_strings), self.embedding_dim))

        for i, vec in enumerate(vectors_as_strings):
            vector = np.fromstring(vec, sep=" ")
            x_val[i] = vector

        return x_val

    def show_activations_of_vector(self,
                                   vectors_as_strings=[],
                                   translations=[],
                                   language="",
                                   max_highlights=3):
        """Plot the activations of an instance.

        Args:
            vectors_as_strings: list(str)
                                The vectors as given by a file, in string
                                format, e.g. "0.2 0.12 0.54 ..."
            translations:  list
                        A list of instances whose activations should be
                        plotted.
            max_highlights: int
                            The number of labeled instances.
        """
        data = self.prepare_vectors(vectors_as_strings)
        print("shape input: {}".format(data.shape))
        predictions = self.model.predict(data)

        self.visualizer.display_concept_activations_without_cosine(
            instances=translations,
            classes=self.concepts,
            predictions=predictions,
            max_highlights=max_highlights,
            language=language)

    def show_contextualized_activations(self,
                                        sentences,
                                        word,
                                        max_highlights=5):
        """Show the contextualized activation profile of a word.

        Args:
            sentences:  list(str)
                        A list of sentences providing the context for the word
                        whose activations are to be visualized.
            word:   str
                    The word whose activations are to be visualized.
            max_highlights: int
                            The number of labeled concepts in the activation
                            profile.
        """
        # check if instance is in training or test data
        global_embedding = self.embedding.get_embedding_for_word(word).reshape(
            (1, self.embedding_dim))

        # get a dict with all occurrences of the given word/phrase per
        # sentence, collected in a list
        context_embeddings = self.embedding.get_contextualized_embeddings(
            sentences=sentences, word=word)

        assert global_embedding.shape == (1, self.embedding_dim), (
            "Please make sure the model gets the correct embedding dimensions "
            "(1, {}), currently the shape is {}.".format(
                self.embedding_dim, global_embedding.shape))
        # assert context_embeddings.shape == (1, self.embedding_dim), (
        #  "Please make sure the model gets the correct embedding dimensions "
        #     "(1, {}), currently the shape is {}.".format(
        #         self.embedding_dim, context_embeddings.shape))

        print("shape global_embedding: {}".format(global_embedding.shape))
        global_prediction = self.model.predict(global_embedding)

        contextualized_predictions = []

        # iterate over all embeddings per sentence
        for s, context_embedding in enumerate(context_embeddings):
            predictions_per_sentence = {}
            for position, embedding in context_embedding.items():
                print("embedding shape: {}".format(embedding.shape))
                pred = self.model.predict(
                    embedding.reshape((1, self.embedding_dim)))
                predictions_per_sentence[position] = pred

            predictions_per_sentence["global"] = global_prediction

            contextualized_predictions.append(predictions_per_sentence)

            self.visualizer.display_contextualized_concept_activations(
                sentence=sentences[s],
                word=word,
                classes=self.concepts,
                predictions=predictions_per_sentence,
                max_highlights=max_highlights)

    def train(self, live_plotting=False, inspect_concept=False):
        """Train the NVC model.

        Args:
            inspect_concept:    bool or str
                                Inspect a given concept with all instances.
        """
        self.create_model_name_and_dir()
        # create two matrices: one for the instances and one for the concepts
        embed_matrix, label_matrix = self.create_embedding_and_label_matrix(
            embedding_dim=self.embedding_dim)

        print("shape embedding matrix: {}".format(embed_matrix.shape))
        print("shape label matrix: {}\n".format(label_matrix.shape))
        # create a callback for metrics like F1 score
        metrics_callback = MetricsCallback(
            labels=self.concepts, threshold=self.config["classif_threshold"])
        callbacks = [metrics_callback]

        # configure early stopping if it is set in the config file
        if self.config["early_stopping"]:
            early_stop = EarlyStopping(
                monitor="val_categorical_accuracy",  # "val_loss",
                min_delta=self.config["min_delta"],  # 0.0002
                patience=self.config["patience"],  # 3
                verbose=1,
                mode='auto',
                baseline=None,
                restore_best_weights=False)

            callbacks.append(early_stop)

        if live_plotting:
            if live_plotting_available:
                callbacks.append(PlotLossesCallback())
            else:
                print("Please install the live plotting library for Keras via "
                      "pip install livelossplot "
                      "[https://github.com/stared/livelossplot/]")

        if self.config["cross_val"]:
            # Train the model via cross validation and save the best one
            self.train_with_cross_validation(
                embed_matrix, label_matrix, callbacks, save_best_model=True)

        else:
            # otherwise, split the data for conventional training
            (x_train, y_train, x_dev, y_dev, idx_dev, x_val, y_val,
             test_instances_list) = self.split_data(
                 embed_matrix,
                 label_matrix,
                 train_size=self.config["train_set_size"])

            # TODO: this is redundant
            # x_train = np.expand_dims(x_train, axis=2)
            # x_dev = np.expand_dims(x_dev, axis=2)
            # if the class weights option is set in the config file,
            # calculate the weights from the given training labels
            d_class_weights = None

            if self.config["use_class_weights"]:
                print("\nCalculating class weights.")
                class_weights = class_weight.compute_class_weight(
                    "balanced", np.unique(np.asarray(y_train)),
                    np.asarray(y_train).flatten())

                d_class_weights = dict()
                for idx in range(len(y_train[0])):
                    d_class_weights[idx] = class_weights[0]
                    d_class_weights[len(y_train[0]) + idx] = class_weights[1]

                self.results_to_write["use_class_weights"] = ":".join(
                    [str(round(x, 2)) for x in class_weights])

                print("Done.")

            # build and compile the model
            print("\nTraining new model: '{}'.\n".format(self.model_name))
            # the number of output units corresponds to the number of concepts
            model = self.get_model(num_output_units=label_matrix.shape[1])

            model.summary()

            # train the model with the configuration as given in the config
            # file
            model.fit(
                x_train,
                y_train,
                validation_data=(x_dev, y_dev),
                epochs=self.config["epochs"],
                callbacks=callbacks,
                batch_size=self.config["batch_size"],
                class_weight=d_class_weights,
                verbose=True)

            self.model = model

            # if a specific concept is given for manual inspection,
            # print the predicted and true labels to the screen
            if inspect_concept:
                self.inspect_predictions_for_class(inspect_concept, x_dev,
                                                   idx_dev, y_dev)

            if not self.debug:
                # save the model and the configuration
                self.model.save(self.model_name)
                utils.save_config_to_file(self.config, self.model_path,
                                          self.details_str)
            # evaluate the model and save the evaluation data to a file
            print("Evaluate on the validation set:\n")
            self.predict_and_evaluate(
                x_val=x_val, y_val=y_val, test_instances=test_instances_list)

            if not self.debug:
                utils.save_results(self.results_file, self.results_to_write)

    def load_pretrained_model(self,
                              trained_model="",
                              x_val_file="",
                              x_val_vec=[],
                              determined_threshold=0.5,
                              save_results=True):
        """Load a pre-trained model.

        Args:
            trained_model:  str
                            The path to the h5 model.
            x_val_file:     str
                            The path to the validation data.

        """
        if not trained_model:
            print("Please specify a model.\nExit.")
            sys.exit(1)

        self.pretrained = True
        self.results_to_write["pretrained"] = self.pretrained
        # Create the original embedding and label matrix (with *all* data)
        # These matrices always have the same ordering when given the same
        # filtered_data json file.
        # embed_matrix, label_matrix = self.create_embedding_and_label_matrix(
        #     embedding_dim=self.embedding_dim)

        print("Loading pretrained model: '{}'\n".format(trained_model))
        self.model = load_model(trained_model)
        path = trained_model.split("/")
        # overwrite model path
        self.model_path = "/".join(path[:-1])
        self.details_str = path[-1].replace("model_", "").replace(".h5", "")

        self.results_to_write["model_name"] = path[-1]

        self.visualizer.update_paths(
            model_path=self.model_path,
            details_str=self.details_str,
            pretrained=self.pretrained)

        if x_val_file:
            (embed_matrix,
             label_matrix) = self.create_embedding_and_label_matrix(
                 embedding_dim=self.embedding_dim)
            test_instances_list = []
            # create a label and embedding matrix from the given test data
            # file
            with open(x_val_file, "r") as read_handle:
                lines = read_handle.readlines()
                if self.embedding.embedding_model_name == "bert":
                    x_val = np.zeros((432702, self.embedding_dim))
                    y_val = np.zeros((432702, len(self.concepts)))
                else:
                    x_val = np.zeros((len(lines), self.embedding_dim))
                    y_val = np.zeros((len(lines), len(self.concepts)))
                    
                for i, line in enumerate(lines):
                    if self.embedding.embedding_model_name == "bert":
                        instance = line.split("\t")[0]
                    else:
                        instance = line.split()[0]
                    test_instances_list.append(instance)
                    
                    idx_of_instance = self.word2id[instance]

                    # create the validation matrices
                    x_val[i] = embed_matrix[idx_of_instance]
                    y_val[i] = label_matrix[idx_of_instance].todense()

            self.threshold = determined_threshold

            # predict the labels for the given test data and evaluate the
            # model's performance
            results_per_class, avrg_results = self.predict_and_evaluate(
                x_val=x_val, y_val=y_val, test_instances=test_instances_list)

            self.results_to_write["weighted_f1_old_ts_best_model"] = 0.0
            self.results_to_write["weighted_f1_old_ts_average"] = 0.0

            if not self.debug:
                if save_results:
                    utils.save_results(self.results_file,
                                       self.results_to_write)

            return results_per_class, avrg_results

    def split_data_for_cross_val(self, embedding_matrix, label_matrix):
        """Split the data in train/dev and test set."""
        label_matrix = label_matrix.toarray()

        # create a 10% test set separate from the cross validation splits
        num_test_instances = int(0.1 * len(embedding_matrix)) + 2
        assert num_test_instances > 0, "No test instances found."

        rand_start = np.random.randint(
            0,
            len(embedding_matrix) - num_test_instances)
        test_indices = list(range(rand_start, rand_start + num_test_instances))

        # get the actual test instances
        test_instances = []
        for idx in test_indices:
            test_instances.append(self.id2word[idx])

        train_and_val_embed_matrix = np.zeros(
            (len(embedding_matrix) - num_test_instances, self.embedding_dim))
        train_and_val_label_matrix = np.zeros(
            (len(embedding_matrix) - num_test_instances, len(self.concepts)))
        test_embed_matrix = np.zeros((num_test_instances, self.embedding_dim))
        test_label_matrix = np.zeros((num_test_instances, len(self.concepts)))

        assert (len(train_and_val_embed_matrix) +
                len(test_embed_matrix) == len(embedding_matrix))

        self.results_to_write["num_train_val_instances"] = len(
            train_and_val_embed_matrix)
        self.results_to_write["num_test_instances"] = len(test_embed_matrix)

        t = 0
        tv = 0
        for i, row in enumerate(embedding_matrix):
            if i in test_indices:
                test_embed_matrix[t] = embedding_matrix[i]
                test_label_matrix[t] = label_matrix[i]
                t += 1
            else:
                train_and_val_embed_matrix[tv] = embedding_matrix[i]
                train_and_val_label_matrix[tv] = label_matrix[i]
                tv += 1

        print("test_embed_matrix: {}".format(test_embed_matrix.shape))
        print("test_label_matrix: {}".format(test_label_matrix.shape))
        print("train_and_val_embed_matrix: {}".format(
            train_and_val_embed_matrix.shape))
        print("train_and_val_label_matrix: {}".format(
            train_and_val_label_matrix.shape))

        return (train_and_val_embed_matrix, train_and_val_label_matrix,
                test_embed_matrix, test_label_matrix, test_instances)

    def train_with_cross_validation(self,
                                    embedding_matrix,
                                    label_matrix,
                                    callbacks,
                                    save_best_model=True,
                                    show_random_baseline=True):
        """Run the training with cross validation.

        Args:
            embedding_matrix:   numpy ndarray
                                The data matrix.
            label_matrix:   numpy ndarray
                            The label matrix.
            callbacks:      list
                            Callbacks like metrics and early stopping.
            save_best_model:    bool
                                If True, save the best model as h5 file.
        """
        # split the data in train/dev and separate validation set:
        (train_and_val_embed_matrix, train_and_val_label_matrix,
         test_embed_matrix, test_label_matrix,
         test_instances) = self.split_data_for_cross_val(
             embedding_matrix, label_matrix)

        # if the config says 'cross_val: True' but not the number of folds,
        # take 5 folds
        if not self.config["cross_val"] > 1:
            n_splits = 5
        else:
            n_splits = self.config["cross_val"]

        all_f1s = []

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        best_f1 = -1.0

        print("\nCalculating class weights.")

        class_weights = compute_class_weight(
            class_weight="balanced",
            y=train_and_val_label_matrix.flatten(),
            classes=np.unique(train_and_val_label_matrix))
        print(class_weights)
        print("Done.\n")

        for i, (train_idx, dev_idx) in enumerate(
                kf.split(train_and_val_embed_matrix,
                         train_and_val_label_matrix)):

            x_train_cv = train_and_val_embed_matrix[train_idx]
            y_train_cv = train_and_val_label_matrix[train_idx]

            # calculate class weights if specified in config file
            class_weights = None
            if self.config["use_class_weights"]:
                print("\nCalculating class weights.")

                print("shape y_train_cv: {}".format(y_train_cv.shape))

                class_weights = compute_sample_weight(
                    class_weight="balanced",
                    y=np.asarray(y_train_cv),
                    indices=None)
                print("Done.\n")
            # create the dev data
            x_dev_cv = train_and_val_embed_matrix[dev_idx]
            y_dev_cv = train_and_val_label_matrix[dev_idx]  # .todense()

            # build and compile the model
            model = self.get_model(
                num_output_units=train_and_val_label_matrix.shape[1])

            if i == 0:
                model.summary()

            print("\nTraining on fold {} ...\n".format(i + 1))
            # train the model
            model.fit(
                x_train_cv,
                y_train_cv,
                validation_data=(x_dev_cv, y_dev_cv),
                epochs=self.config["epochs"],
                batch_size=self.config["batch_size"],
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=True)

            # save best model wrt weighted F1 score
            current_f1 = callbacks[0].report['weighted avg']["f1-score"]
            all_f1s.append(current_f1)

            if current_f1 > best_f1:
                best_f1 = current_f1
                self.model = model
                data = x_dev_cv
                # dev_idx_to_save = dev_idx
                labels = y_dev_cv
                # train_idx_to_save = train_idx

        # get predictions of best model and optimize threshold
        best_model_predictions = self.model.predict(data)
        # roc/auc not really useful for unbalanced data
        # compute_roc_auc(labels, best_model_predictions, self.concepts)
        # # plot prec/rec curve with old threshold
        # plot_precision_recall_curve(
        #     y_pred=best_model_predictions,
        #     y_true=labels,
        #     threshold=self.threshold)

        self.threshold = determine_best_threshold(
            current_f1=best_f1, y_pred=best_model_predictions, y_true=labels)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("\nPredicting and evaluating with best model on "
              "separate test set with threshold of {:.2f}:".format(
                  self.threshold))
        self.predict_and_evaluate(
            test_embed_matrix,
            test_label_matrix,
            test_instances=test_instances,
            save_predictions=True)

        avrg_f1 = np.sum(all_f1s) / n_splits

        print("\nAverage F1 of all {} splits (threshold: 0.5): {}.\n"
              "Best model achieved F1 of {} (threshold: 0.5).\n".format(
                  n_splits, avrg_f1, best_f1))

        self.results_to_write["weighted_f1_old_ts_best_model"] = best_f1
        self.results_to_write["weighted_f1_old_ts_average"] = avrg_f1
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        if save_best_model:
            if not self.debug:
                print("Saving best model.")
                self.model.save(self.model_name)
                utils.save_results(self.results_file, self.results_to_write)

        if show_random_baseline:
            clf = DummyClassifier(
                strategy="stratified", random_state=None, constant=None)

            clf.fit(X=train_and_val_embed_matrix, y=train_and_val_label_matrix)

            predictions = clf.predict(test_embed_matrix)

            predictions[predictions >= self.threshold] = 1
            predictions[predictions < self.threshold] = 0
            # calculate precision, recall and F1 score
            (results_per_class_table,
             average_results_table) = self.calculate_scores(
                 test_label_matrix, predictions)

            print(
                "\nRandom classifier results per concepts (threshold: {:.2f}):\n".
                format(self.threshold))
            print(results_per_class_table)
            print("\nRandom classifier average results (threshold: {:.2f}):\n".
                  format(self.threshold))
            print(average_results_table)

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def predict_single_vector(self, input_vec=[], concepts=[]):
        """Predict a single given vector."""
        # let the model predict
        predictions = self.model.predict(input_vec)
        data = {"prediction": predictions[0], "concepts": concepts}
        df = pd.DataFrame(data)

        # assign label via a threshold
        # predictions[predictions >= self.threshold] = 1
        # predictions[predictions < self.threshold] = 0
        sorted_activations = df.sort_values(by=["prediction"], ascending=False)

        return sorted_activations

    def predict_and_evaluate_translations(self,
                                          datafile,
                                          save_predictions=False):
        """Predict the concepts for translations of English instances."""
        # create list of data dicts
        data = []
        # structure: english token, chinese token, chinese_vector
        with open(datafile, "r") as jsonl_file:
            for line in jsonl_file:
                d = json.loads(line)
                if d["vector"]:
                    data.append(d)
        # create x_val
        x_val = np.zeros((len(data), self.embedding_dim))
        y_val = np.zeros((len(data), len(self.concepts)))
        for i, instance_dict in enumerate(data):
            x_val[i] = instance_dict["vector"]
            # get the id of the current instance in the data matrix
            idx = self.inst2id[instance_dict["instance"]]
            # get the concept vector for the given (English) instance
            reps = self.filtered_data_matrix[idx].todense()
            y_val[i] = reps

        print("#translated instances: {}\n#concepts:{}\n".format(
            len(data), len(self.concepts)))
        print(
            "\nDone.\nAvrg #positive concepts per instance (translations): {}".
            format(np.count_nonzero(y_val) / len(data)))

        avrg_instances_per_concept = sum(np.count_nonzero(
            y_val, axis=0)) / len(y_val[0, :])
        print("Avrg #instances per concept (translations): {}".format(
            avrg_instances_per_concept))
        # let the model predict
        predictions = self.model.predict(x_val)
        # assign label via a threshold
        predictions[predictions >= self.threshold] = 1
        predictions[predictions < self.threshold] = 0

        # calculate precision, recall and F1 score
        (results_per_class_table,
         average_results_table) = self.calculate_scores(y_val, predictions)

        print("\nResults per concept (translated vectors, threshold: {:.2f}):".
              format(self.threshold))
        print(results_per_class_table)

        print(
            "\nAverage results for translated vectors (threshold: {:.2f}):\n".
            format(self.threshold))
        print(average_results_table)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        return results_per_class_table, average_results_table

    def predict_and_evaluate(self,
                             x_val,
                             y_val,
                             test_instances,
                             save_predictions=True):
        """Predict the labels for a given dataset."""
        # let the model predict
        predictions = self.model.predict(x_val)
        # assign label via a threshold
        predictions[predictions >= self.threshold] = 1
        predictions[predictions < self.threshold] = 0

        # calculate precision, recall and F1 score
        (results_per_class_table,
         average_results_table) = self.calculate_scores(y_val, predictions)
        print("\nResults per concept (english vectors, threshold: {:.2f}):\n".
              format(self.threshold))
        print(results_per_class_table)

        print("\nAverage results (threshold: {:.2f}):\n".format(
            self.threshold))
        print(average_results_table)

        # save both the predictions and the used test data (the words) to
        # text files.
        if save_predictions:

            with open(
                    "{}/predictions_{}.csv".format(self.model_path,
                                                   self.details_str),
                    "w") as write_handle:
                for i, instance in enumerate(test_instances):
                    write_handle.write("{}\t{}\n".format(
                        instance, "\t".join([str(x) for x in predictions[i]])))

            with open(
                    "{}/ground_truth_{}.csv".format(self.model_path,
                                                    self.details_str),
                    "w") as write_handle:
                for i, instance in enumerate(test_instances):
                    write_handle.write("{}\t{}\n".format(
                        instance, "\t".join([str(x) for x in y_val[i]])))
        return results_per_class_table, average_results_table

    def calculate_scores(self, y_val, predictions):
        """Calculate precision, recall and F1 score.

        Args:
            y_val:  numpy ndarray
                    The validation data.
            predictions:    numpy ndarray
                            The predicted activations.
        """
        # -------------------- Scores per concept ----------------------
        # calculate scores for all classes
        precs, recs, f1_scores, supports = precision_recall_fscore_support(
            y_val, predictions, labels=range(len(self.concepts)))

        # create a dataframe for all classes
        col_names = ["concept", "precision", "recall", "F1", "support"]
        results_per_class_table = pd.DataFrame(
            list(zip(self.concepts, precs, recs, f1_scores, supports)),
            columns=col_names)

        # -------------------- Averaged scores -------------------------
        # calculate averaged scores
        f1_weighted = f1_score(y_val, predictions, average="weighted")
        f1_macro = f1_score(y_val, predictions, average="macro")
        f1_micro = f1_score(y_val, predictions, average="micro")

        precision_weighted = precision_score(
            y_val, predictions, average="weighted")
        precision_macro = precision_score(y_val, predictions, average="macro")
        precision_micro = precision_score(y_val, predictions, average="micro")

        recall_weighted = recall_score(y_val, predictions, average="weighted")
        recall_macro = recall_score(y_val, predictions, average="macro")
        recall_micro = recall_score(y_val, predictions, average="micro")

        # create a dataframe for the averaged scores
        col_names_avrg = ["score", "weighted", "macro", "micro"]
        average_results_table = pd.DataFrame(
            list(
                zip(["Precision", "Recall", "F1"],
                    [precision_weighted, recall_weighted, f1_weighted],
                    [precision_macro, recall_macro, f1_macro],
                    [precision_micro, recall_micro, f1_micro])),
            columns=col_names_avrg)

        self.results_to_write["determined_threshold"] = self.threshold
        self.results_to_write["f1_weighted"] = f1_weighted
        self.results_to_write["f1_macro"] = f1_macro
        self.results_to_write["f1_micro"] = f1_micro

        self.results_to_write["precision_weighted"] = precision_weighted
        self.results_to_write["precision_macro"] = precision_macro
        self.results_to_write["precision_micro"] = precision_micro

        self.results_to_write["recall_weighted"] = recall_weighted
        self.results_to_write["recall_macro"] = recall_macro
        self.results_to_write["recall_micro"] = recall_micro

        return results_per_class_table, average_results_table

    def compare_random_samples(self,
                               file_in_fasttext_format,
                               num_samples=100,
                               top_n=10,
                               measure="spearman",
                               seed=42):
        """Calculate the ranking correlation between randomly sampled vectors.

        Args:
            file_in_fasttext_format:    str
                                        Format: instance vector
            num_samples:    int
                            Number of vectors to be sampled.
            top_n:          int
                            Number of top concepts to compare.
            measure:        str
                            The measure with which the correlation is
                            calculated.
        """
        np.random.seed(seed)

        # open the .vec file and retrieve the instances and their vectors
        with open(file_in_fasttext_format, "r") as read_handle:
            # extract the number of instances and dimensions from the header
            num_instances, dims = map(int, read_handle.readline().split())
            print("#instances in file: {}\n#dimensions: {}\n".format(
                num_instances, dims))

            indices = [i for i in range(0, num_instances)]
            np.random.shuffle(indices)
            random_start = np.random.randint(0, num_instances)
            random_indices = indices[random_start:random_start + num_samples]

            # create a data dict of instances and vectors
            data = {}
            indices = []
            for i, line in enumerate(read_handle):
                if i in random_indices:
                    tokens = line.rstrip().split(' ')
                    data[tokens[0]] = np.array(tokens[1:]).astype(float)
                    # indices = [i for i in range(0, len(vocab))]
                    indices.append(i)

        vocab = list(data.keys())
        assert len(indices) == len(vocab)

        # prepare the input data
        x_val = np.zeros((num_samples, self.embedding_dim))

        random_instances = []

        p = 0
        for instance, vector in data.items():
            # instance = vocabulary[rand_idx]
            random_instances.append(instance)
            x_val[p] = vector
            p += 1

        print("random instances: {}".format(random_instances))
        print("shape input: {}".format(x_val.shape))
        assert x_val.shape[0] == len(random_instances), (
            "Shape of input and number of random instances do not match.")

        # predict the concept activations
        predictions = self.model.predict(x_val)

        instances_and_max_concepts = {}

        for w, prediction in enumerate(predictions):
            print("shape prediction: {}".format(prediction.shape))

            top_concepts = self.visualizer.get_top_n_concepts(
                prediction, concepts=self.concepts, top_n=top_n)

            instances_and_max_concepts[random_instances[w]] = top_concepts

        return instances_and_max_concepts


if __name__ == '__main__':

    from models.embeddings import Embedding

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "raw_data", type=str, help=("The path to the tsv data."))

    parser.add_argument(
        "embedding_file", type=str, help=("The path to the embedding data."))

    ARGS, _ = parser.parse_known_args()

    embedding = Embedding(embedding_file=ARGS.embedding_file, voc_limit=100000)

    model = NeuralVectorConceptualizer(
        config_file=ARGS.config, embedding=embedding)
    model.train(inspect_concept=False)
