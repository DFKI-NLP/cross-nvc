#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocessing the Microsoft Concept Graph (MCG) corpus.

A script for preparing the basic corpus such that is easy to modify
by just adding new preprocessing functions to a map function.

@author lisa.raithel@dfki.de
"""

import json
import numpy as np
import pandas as pd
import sys
import time

from collections import defaultdict

from sklearn.feature_extraction import DictVectorizer


class DataLoader(object):
    """Class for loading and preparing data for a model."""

    def __init__(self,
                 embedding,
                 min_count_of_instances,
                 use_logs=True,
                 max_length_of_instances=False,
                 check_concepts_for_cosine=False,
                 debug=False):
        """Initialize the loader.

        Args:
            embedding:  class
                        A class providing embedding utilities.
            use_logs    bool
                        Use the logarithm of the MCG scores.
            max_length_of_instances:   int/bool
                            The maximum length an instance should have.
        """
        self.filtered_data = defaultdict(lambda: defaultdict(float))
        self.use_logs = use_logs
        self.max_length_of_instances = max_length_of_instances
        self.embedding = embedding
        self.raw_data_dict = defaultdict(lambda: defaultdict(float))
        self.timestr = time.strftime("%Y_%m_%d-%H_%M")
        self.debug = debug
        self.check_concepts_for_cosine = check_concepts_for_cosine
        self.min_count_of_instances = min_count_of_instances

    def load_raw_data(self,
                      path_to_data="",
                      header=[],
                      chunk_size=1000**2,
                      save_to_json=True,
                      selected_concepts=[]):
        """Prepare the raw data in dict format.

        Args:
            path_to_data:   str
                            The path to the original TSV data or a JSON
                            dump of it.
            header: list
                    The header of the TSV file.
            chunk_size: int
                        The chunk size for reading in the TSV.
            save_to_json:   bool
                            Save the resulting dict to a JSON file.
        Returns:
            A dict of all concepts associated with their instances:
            raw_data = {concept_1: {instance_1: rep_1, instance_2: rep_2},
                        concept_2: {instance_2: rep_2, instance_3, rep_3},...}
        """
        if path_to_data.endswith(("json", "JSON")):
            try:
                # if there is already a JSON file provided, use it instead
                # of reading in the original data (slow)
                with open(path_to_data, "r") as json_file:
                    print("Loading data file '{}'.\n".format(path_to_data))
                    self.raw_data_dict = json.load(json_file)

            except FileNotFoundError as e:
                print("File '{}' not found, please provide the correct "
                      "path.\n{}".format(path_to_data, e))
                sys.exit(1)
            # if there is only a TSV file provided, create a dict for all
            # concepts and instances and calculate the REP scores
            except json.decoder.JSONDecodeError as e:
                print("JSON file '{}' invalid: {}".format(path_to_data, e))
                sys.exit(1)

        else:
            concepts_not_in_embedding = set()
            instances_not_in_embedding = set()
            try:
                print("Preparing raw data.")
                if self.use_logs:
                    print("Using log of REP values.\n")
                # read in data chunk-wise
                for c, chunk in enumerate(
                        pd.read_csv(
                            path_to_data,
                            chunksize=chunk_size,
                            delimiter="\t",
                            names=header)):
                    df = pd.DataFrame(chunk)

                    for i, row in df.iterrows():
                        if i % 100000 == 0:
                            print(
                                "{} lines done, #concepts not in embedding: {}"
                                ", #instances not in embedding: {},"
                                " #collected concepts: {}".format(
                                    i, len(concepts_not_in_embedding),
                                    len(instances_not_in_embedding),
                                    len(self.raw_data_dict)),
                                end="\r",  # noqa: E901
                                flush=True)  # noqa: E901

                        # if int(row["count"]) < self.min_count_of_instances:
                        #    continue

                        if selected_concepts:
                            if not row["concept"] in selected_concepts:
                                continue
                        # check if a concept word or phrase is in the embedding
                        # vocabulary (checks also modified versions of the
                        # word, like upper/lower case, with underscores etc.)
                        # [the modified concepts are actually only necessary
                        # for cosine similarity comparisons, but checking it
                        # now makes for a better work flow later]
                        concept = str(row["concept"])

                        if self.check_concepts_for_cosine:
                            modified_concept = self.embedding.in_embedding(
                                concept, self.max_length_of_instances)

                            if not modified_concept:
                                concepts_not_in_embedding.add(
                                    str(row["concept"]))
                                continue
                            else:
                                concept = modified_concept
                        # concept = row["concept"]
                        proba_c_given_e = float(row["p(c|e)"])
                        proba_e_given_c = float(row["p(e|c)"])
                        instance = str(row["instance"])
                        # check if a version of the current instance is in the
                        # embedding vocab
                        # (this could also be a shorter phrase than the
                        # original instance)
                        # if instance == "confucius":
                        #     continue
                        modified_instance = self.embedding.in_embedding(
                            word=instance,
                            max_length_of_instances=self.
                            max_length_of_instances)

                        # if none of the modified instances is in the embedding
                        # vocabulary, go to the next instance
                        if not modified_instance:
                            instances_not_in_embedding.add(instance)
                            continue
                        else:
                            instance = modified_instance

                        if self.use_logs:
                            # use natural logarithm to calculate the REP values
                            self.raw_data_dict[concept][instance] = np.log(
                                proba_c_given_e) + np.log(proba_e_given_c)

                        else:
                            new_rep = proba_c_given_e * proba_e_given_c
                            self.raw_data_dict[concept][instance] = new_rep

                with open(
                        "concepts_not_in_embedding_{}.txt".format(
                            self.embedding.embedding_model_name),
                        "w") as write_handle:
                    for concept in concepts_not_in_embedding:
                        write_handle.write(concept)
                        write_handle.write("\n")
                with open(
                        "instances_not_in_embedding_{}.txt".format(
                            self.embedding.embedding_model_name),
                        "w") as write_handle:
                    for instance in instances_not_in_embedding:
                        write_handle.write(instance)
                        write_handle.write("\n")

                # save the data to a JSON file to increase processing speed
                # for the next run
                if save_to_json:

                    if selected_concepts:
                        file_name = "selected_raw_data_dict_{}_maxlen{}".format(
                            self.embedding.embedding_model_name,
                            self.max_length_of_instances)
                    else:
                        file_name = "raw_data_dict_{}_maxlen{}".format(
                            self.embedding.embedding_model_name,
                            self.max_length_of_instances)

                    if self.check_concepts_for_cosine:
                        file_name += "_cc.json"
                    else:
                        file_name += ".json"

                    self.save_to_json(
                        data=self.raw_data_dict, file_name=file_name)

            except FileNotFoundError as e:
                print("File '{}' not found, please provide the correct "
                      "path.\n{}".format(path_to_data, e))
                sys.exit(1)

        print("\n#concepts in raw data: {}\n".format(
            len(self.raw_data_dict.keys())))

        return self.raw_data_dict

    def load_filtered_data_matrix(self,
                                  data=None,
                                  min_num_instances=2,
                                  min_rep=-6.0,
                                  save_to_json=True,
                                  selected_concepts=[]):
        """Load the filtered data.

        Args:
            data:   dict or str
                    The data as dict or JSON file -- could be either the raw
                    data or the already filtered data.
            min_num_instances:  int
                                The minimum number of instances a concept
                                need to have.
            min_rep:    float
                        The minimum REP value an instance needs to have.
            save_to_json:   bool
                            Save the filtered data to a JSON file.
            selected_concepts:  list
                                A list of selected concepts that should be
                                considered (and no other concepts).
        """
        filtered_data = defaultdict(lambda: defaultdict(float))
        # try if the given file is already filtered (because then it's in JSON
        # format)
        try:
            with open(data, "r") as json_file:
                print("Loading existing filtered data file '{}'.".format(data))
                filtered_data = json.load(json_file)
        # otherwise, check if the filtered data already exists
        except json.decoder.JSONDecodeError as e:
            print("JSON DecodingError: {}.\nPlease provide a valid JSON file.".
                  format(e))
            sys.exit()

        except TypeError:
            print("Filtering data ...")
            if selected_concepts:
                filtered_data = self.get_selected_concepts_and_instances(
                    data, selected_concepts, min_rep, min_num_instances,
                    save_to_json)
            else:
                filtered_data = self.get_filtered_data(
                    data, min_rep, min_num_instances, save_to_json)

        except FileNotFoundError as e:
            print("No existing filtered data found: {}.\n".format(e))
            sys.exit(1)

        # - create dataframe to get instances for rows and concepts
        #   for columns
        # - the dataframe-from-dict format ensures that every instance is
        #   occurring only *once* in the data matrix, ensuring again that
        #   the train/dev/test splits are disjoint
        # make sure the columns are always filtered lexicographically to
        # ensure the order of labels is always the same for the same data
        # (actually, that doesn't matter)
        # print("Sorting data.")
        # df_filtered.reindex(sorted(df_filtered.columns), axis=1)
        # print("shape filtered data (instances x concepts): {}".format(
        #     df_filtered.shape))
        # print(df_filtered)
        # return df_filtered
        # return the data matrix and one list for the instances, as
        # ordered in the matrix and one list for concepts, as ordered in
        # the matrix
        return self.create_data_matrix(filtered_data=filtered_data)

    def get_selected_concepts_and_instances(self, raw_data_dict,
                                            selected_concepts, min_rep,
                                            min_inst, save_to_json):
        """Get only a predefined set of concepts and their instances.

        Args:
            raw_data_dict:  dict
                            The raw data in dict format.
            selected_concepts:  list
                                A list of concepts.
            min_rep:    float
                        The minimum REP value a instance needs to have.
            min_inst:   int
                        The minimum number of instances a concepts needs
                        to have.
            save_to_json:   bool
                            Save the filtered data to a JSON file.
        Returns:
            The filtered data.
        """
        concept_counter = 0
        filtered_data = defaultdict(lambda: defaultdict(float))

        for concept in selected_concepts:
            concept_counter += 1
            print(
                "{} concepts done".format(concept_counter),
                end="\r",
                flush=True)

            instance_rep_dict = raw_data_dict[concept]
            inst_values = instance_rep_dict.values()
            num_inst_geq_v = sum([x >= min_rep for x in inst_values])

            new_instances_dict = {}
            # check if there are enough instances with a rep > v
            if num_inst_geq_v >= min_inst:
                for instance, rep in instance_rep_dict.items():
                    # take only instances with a REP > v
                    if rep >= min_rep:
                        new_instances_dict[instance] = rep

                filtered_data[concept] = new_instances_dict
            else:
                continue

        if save_to_json:
            file_name = "selected_concepts_i{}_v{}_{}.json".format(
                min_inst, min_rep, self.embedding.embedding_model_name)

            self.save_to_json(data=filtered_data, file_name=file_name)

        return filtered_data

    def get_filtered_data(self, raw_data_dict, min_rep, min_inst,
                          save_to_json):
        """Filter the data.

        Args:
            raw_data_dict:  dict
                            The raw data in dict format.
            min_rep:    float
                        The minimum REP value a instance needs to have.
            min_inst:   int
                        The minimum number of instances a concepts needs
                        to have.
            save_to_json:   bool
                            Save the filtered data to a JSON file.
        Returns:
            The filtered data.
        """
        concept_counter = 0
        filtered_data = defaultdict(lambda: defaultdict(float))

        for concept, instance_rep_dict in raw_data_dict.items():
            concept_counter += 1
            if concept_counter % 10000 == 0:
                print(
                    "{} concepts done".format(concept_counter),
                    end="\r",
                    flush=True)

            # get the REP values of the current instances
            inst_values = instance_rep_dict.values()
            # get the number of instances that have a REP greater or
            # equal than v
            num_inst_geq_v = sum([x >= min_rep for x in inst_values])

            new_instances_dict = {}
            # check if there are enough instances with a rep > v
            if num_inst_geq_v >= min_inst:
                for instance, rep in instance_rep_dict.items():
                    # take only instances with a REP > v
                    if rep >= min_rep:
                        new_instances_dict[instance] = rep

                filtered_data[concept] = new_instances_dict
            else:
                continue

        assert len(filtered_data.keys()) != 0, ("\nNo concepts collected when "
                                                "filtering data.\nPlease "
                                                "change the filter parameters."
                                                "\n")

        if save_to_json:
            file_name = "filtered_data_i{}_v{}_{}_maxlen{}".format(
                min_inst, min_rep, self.embedding.embedding_model_name,
                self.max_length_of_instances)
            if self.check_concepts_for_cosine:
                file_name += "_cc.json"
            else:
                file_name += ".json"

            self.save_to_json(data=filtered_data, file_name=file_name)

        return filtered_data

    def get_all_words_not_in_mcg(self, instances, file_name):
        """Get all words that are not in the concept graph.

        Retrieve all words that are in the vocabulary of word2vec,
        but not in the MCG, and save them to a text file, one word per line.
        These can be used to find instances for plotting etc., which were not
        used in training, validating or testing the model.

        Args:
            filtered_data_matrix:   numpy ndarray
                                    The already filtered data.
            min_inst:   int
                        The minimum number of instances a concepts needs
                        to have.
            min_rep:    float
                        The minimum REP value a instance needs to have.
        """
        print("Collecting instances not in embedding vocabulary from data "
              "matrix ...")
        # get all instances from the data matrix
        # 'remove' the instances used in training/validation/test
        remaining_words = set(self.embedding.vocab) - set(instances)

        print("Writing {} instances to file '{}'.".format(
            len(remaining_words),
            file_name,
        ))

        if not self.debug:
            with open(file_name, "w") as write_handle:
                for word in remaining_words:
                    write_handle.write(word)
                    write_handle.write("\n")

            print("Done.\n")
        return len(remaining_words)

    def separate_data_dict(self, data_dict):
        """Generate data as a list of dicts, collect concepts separately.

        Args:
            data_dict:  dict of dicts
                        The concept->instance->REP dicts.
        Returns:
            A list of dicts and a list of concepts.
        """
        instances_list = []
        concepts_list = []
        for concept, instances_dict in data_dict.items():
            instances_list.append(instances_dict)
            concepts_list.append(concept)

        return instances_list, concepts_list

    def create_data_matrix(self, filtered_data):
        """Create a pd dataframe with the DictVectorizer.

        Args:
            filtered_data:  dict of dicts
                            A mapping of concept->instance->REP.
        Returns:
            A transposed matrix (instances x concepts) with binary cells.
        """
        print("Separating concepts and instances.")
        instances_rep_list, concepts_list = self.separate_data_dict(
            data_dict=filtered_data)

        vectorizer = DictVectorizer(dtype=np.float32, sparse=True)

        print("Creating data matrix.")
        matrix = vectorizer.fit_transform(instances_rep_list)
        matrix = matrix.tocsr()
        # column labels == instances
        instance_names = vectorizer.get_feature_names()

        # create a dataframe from the vectorized data
        # df = pd.DataFrame(matrix, index=concepts_list, columns=column_labels)

        # print("shape dataframe: {}".format(df.shape))

        # transpose the dataframe and convert all non-zero values to one
        # df_transposed = df.transpose().astype(bool).astype(float)
        # note, that the column labels are now row labels!
        matrix_transposed = matrix.transpose().astype(bool).astype(float)

        # print("matrix type: {}\nmatrix_transposed[5:10, 5:10]:\n{}".format(
        #     type(matrix_transposed), matrix_transposed[5:10, 5:10]))
        return matrix_transposed, instance_names, concepts_list

    def load_from_json(self, json_file):
        """Load data from json."""
        with open(json_file, "r") as handle:
            data = json.load(handle)
        return data

    def save_to_json(self, data, file_name):
        """Save data to json file."""
        with open(file_name, "w") as handle:
            json.dump(data, handle)

        print("Data stored in {}.\n".format(file_name))


if __name__ == '__main__':
    json_file = sys.argv[1]
    # "graph_as_dict.json"
    with open(json_file, "r") as handle:
        raw_data = json.load(handle)
