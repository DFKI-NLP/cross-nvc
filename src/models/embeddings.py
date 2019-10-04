##!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A class for easier access to embeddings.

- allows to calculate distances and cosine similarities of words and
  vectors.
- allows to check for vocabulary words / modifies words to fit to
  embedding vocabulary. E.g. the word 'New York' is modified to
  'New_York'.

@author lisa.raithel@dfki.de
"""

import gzip
import logging
import numpy as np
import pandas as pd
import re
import sys

from utils import utils

try:
    from gensim.models import KeyedVectors
    word2vec_available = True
except ImportError:
    word2vec_available = False

try:
    if word2vec_available:
        from gensim.scripts.glove2word2vec import glove2word2vec
        glove_available = True
    else:
        glove_available = False
except ImportError:
    glove_available = False

# only necessary for contextualized embeddings
# try:
#     import torch
#     from pytorch_pretrained_bert import BertTokenizer, BertModel
#     pretrained_bert_available = True
# except ImportError:
#     pretrained_bert_available = False

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


class Embedding(object):
    """Class for making several embeddings accessible."""

    def __init__(self,
                 embedding_or_raw_data_file="",
                 voc_limit=None,
                 model_name="word2vec",
                 intersecting_embedding="",
                 config_file=""):
        """Initialize the embedding module.

        Args:
            embedding_or_raw_data_file: str
                        Path to the word vectors.
            voc_limit:  int
                        A restriction on the embedding vocabulary.
            model_name: str
                        The name of the embedding model that is used:
                        'word2vec' or 'bert'.
            embedding_combiner: numpy callable
                                A function to combine two or more (sub-)
                                word vectors.
        """
        # get the configuration
        self.config = utils.read_config(config_file)

        self.embedding_model_name = model_name
        self.max_seq_length = 20
        self.check_other_embedding = self.config["check_other_embedding"]

        if self.embedding_model_name == "word2vec":
            if word2vec_available and embedding_or_raw_data_file:

                self.word_vectors, self.embedding_dim = self.load_word2vec(
                    embedding_or_raw_data_file, voc_limit)

                print(type(self.word_vectors))

                self.vocab = self.word_vectors.vocab

                self.embedding_combiner = None

            elif word2vec_available and not embedding_or_raw_data_file:
                print("No file for embedding data given.")
                sys.exit(1)
            else:
                raise RuntimeError(
                    "Please download the necessary word2vec embedding file.")

        # elif self.embedding_model_name == "bert":
        #     if pretrained_bert_available:
        #         used_bert_model = "bert-base-uncased"

        #         self.word_vectors, self.embedding_dim = self.load_bert(
        #             used_bert_model)

        #         self.vocab = list(self.word_vectors.keys())

        #         self.embedding_combiner = getattr(
        #             self, self.config["embedding_combiner"])()

        #         print("\nNOTE: the chosen embedding combiner is {}\n".format(
        #             self.embedding_combiner))

        #     else:
        #         raise RuntimeError("Please install pytorch and the bert "
        #                            "model via 'pip install "
        #                            "pytorch-pretrained-bert'")

        elif self.embedding_model_name == "glove":
            self.word_vectors, self.embedding_dim = self.load_glove(
                embedding_or_raw_data_file, voc_limit)

            self.vocab = self.word_vectors.vocab

            self.embedding_combiner = None

        elif self.embedding_model_name == "fasttext":
            self.word_vectors, self.embedding_dim = self.load_fasttext(
                embedding_or_raw_data_file, voc_limit)

            self.vocab = self.word_vectors.vocab

            self.embedding_combiner = None

        else:
            assert False, "Embedding model name unknown: '{}'\n".format(
                self.embedding_model_name)

        if self.check_other_embedding == "word2vec":
            self.load_w2v_as_comparison(intersecting_embedding, voc_limit)

        if self.check_other_embedding == "bert":
            used_bert_model = "bert-base-uncased"
            self.load_bert_as_comparison(used_bert_model)

    def load_w2v_as_comparison(self, embedding_file, voc_limit):
        """..."""
        print("Loading Word2Vec as comparison.")
        self.intersecting_embedding, _ = self.load_word2vec(
            embedding_file, voc_limit=voc_limit)

    def load_bert_as_comparison(self, used_bert_model):
        """..."""
        print("Loading BERT as comparison.")
        self.intersecting_embedding, _ = self.load_bert(used_bert_model)

    def sum(self):
        """Define the embedding combiner function."""
        return np.ndarray.sum

    def max(self):
        """Define the embedding combiner function."""
        return np.ndarray.max

    def mean(self):
        """Define the embedding combiner function."""
        return np.ndarray.mean

    def read_examples(self, sentences):
        """Read a list of `InputExample`s from an input file.

        Partially taken from https://github.com/huggingface/
        pytorch-pretrained-BERT/blob/master/examples/extract_features.py

        Args:
            sentences:  list
                        A list of sentences.
        Returns:
            A list of InputExample objects, containing a sequence and an ID.
        """
        examples = []
        unique_id = 0
        for sentence in sentences:
            line = sentence.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                BertInputExample(
                    unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1

        return examples

    def load_bert(self, used_bert_model):
        """Load BERT.

        Returns:
            A dict mapping from a word to an embedding vector.
        """
        self.embedding_model = BertModel.from_pretrained(used_bert_model)
        self.embedding_model.eval()

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained(
            used_bert_model, do_lower_case=True)

        word_vectors = self.embedding_model.embeddings.word_embeddings.weight
        word2embedding = {}

        for i, word_vector in enumerate(word_vectors):
            # get word (ID depends on BERT vocab.txt)
            word = self.tokenizer.ids_to_tokens[i]
            word2embedding[word] = word_vector.detach().numpy()

        # return bert dict and embedding dimension
        return word2embedding, word_vectors.shape[1]
        # encoded_layers, _ = model(tokens_tensor, segments_tensor)

    def get_contextualized_embeddings(self, sentences, word):
        """..."""
        print("sentence: {}\nword: {}\n".format(sentences, word))
        # tokens = self.tokenizer.tokenize(sentence)
        in_sentence = True
        for sentence in sentences:
            if word not in sentence:
                in_sentence = False

        assert in_sentence, (
            "The word '{}' cannot be found in the  sentences: "
            "{}.").format(word, sentences)

        examples = self.read_examples(sentences)
        features = self.convert_examples_to_features(
            examples=examples, seq_length=self.max_seq_length)
        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature

        all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long)

        encoded_layers, _ = self.embedding_model(all_input_ids, all_input_mask)
        assert len(encoded_layers) == 12

        last_layer = encoded_layers[-1]

        embeddings = []
        # get the embedding of the single word with its corresponding context
        for i, feature in enumerate(features):

            sentence = sentences[i]
            sentence_embedding = last_layer[i, :].detach().numpy()
            input_ids = feature.input_ids
            assert len(input_ids) == sentence_embedding.shape[0]

            tokenized_sentence = feature.tokens
            subtokens = self.tokenizer.tokenize(word)

            subtoken_indices = self.get_subtoken_indices(
                tokenized_sentence, subtokens)

            embeddings_for_sentence = {}

            for k, subtoken_idx in enumerate(subtoken_indices):
                word_embeddings_list = []

                for idx in subtoken_idx:

                    # reshape the single word embeddings
                    word_embedding = sentence_embedding[idx, :].reshape(
                        (1, self.embedding_dim))

                    word_embeddings_list.append(word_embedding)
                word_embeddings = np.array(word_embeddings_list)

                context_embeddings = np.concatenate(
                    seq=word_embeddings, axis=0)

                print("shape context embeddings: {}".format(
                    context_embeddings.shape))

                combined_embedding = self.combine_word_vectors(
                    context_embeddings)
                print("shape combined embeddings: {}".format(
                    combined_embedding.shape))

                embeddings_for_sentence[k] = combined_embedding

            embeddings.append(embeddings_for_sentence)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        return embeddings

    def get_subtoken_indices(self, tokenized_sentence, subtokens):
        """Get the indices of the subtokens.

        Since some words are split up in subtokens, we need to find all
        indices of these subtokens in the sentence.
        Args:
            tokens: list of str
                    The tokenized version of a sentence.
            subtokens:  list of str
                        The tokenized version of a word.
        Returns:
            list (of list) of ints
            A list of indices for every occurrence of the subtokens
            in the sentence.
        """
        # find possible starting positions of the target word
        starting_positions = [
            i for i, x in enumerate(tokenized_sentence) if x == subtokens[0]
        ]
        num_subtokens = len(subtokens)

        # if the target word is not split up, just return its index
        if len(subtokens) == 1:

            return [starting_positions]

        all_idx_of_subtokens = []

        # otherwise, iterate over all possible starting positions and
        # check if the following indices/(sub)tokens are part of the
        # target word
        for idx in starting_positions:
            idx_of_subtokens = []
            j = 1
            start = idx
            while j <= num_subtokens - 1:

                if tokenized_sentence[start + j] == subtokens[j]:
                    idx_of_subtokens.append(idx)
                    idx_of_subtokens.append(start + j)
                    j += 1
                    continue
                else:
                    break
            if idx_of_subtokens:
                all_idx_of_subtokens.append(idx_of_subtokens)

        return all_idx_of_subtokens

    def load_glove(self, embedding_file, voc_limit):
        """Prepare the GloVe vectors.

        Returns:
            A dict mapping from a word to an embedding vector.
        """
        print("embedding file: {}\nvoc limit: {}".format(
            embedding_file, voc_limit))
        try:
            # this is the file the converted GloVe vectors are written to
            glove_in_w2v_format_file = "gensim_glove_vectors.txt"

            print("Converting GloVe to Word2Vec format.\n")
            num_vecs, dims = glove2word2vec(
                glove_input_file=embedding_file,
                word2vec_output_file=glove_in_w2v_format_file)

            print("#word vectors: {}, vector dimensions: {}".format(
                num_vecs, dims))
            print("Loading GloVe model.\n")
            glove_model = KeyedVectors.load_word2vec_format(
                glove_in_w2v_format_file, binary=False, limit=voc_limit)

        except ValueError as e:
            print(e)

        print("vector dims: {}, check 'dog': {}".format(
            glove_model.vector_size, glove_model["dog"]))
        # return a word vector dict and the embedding dimension
        return glove_model.wv, glove_model.vector_size

    def load_fasttext(self, embedding_file, voc_limit):
        """Prepare the fastText model and word vectors.

        Returns:
            A dict mapping from a word to an embedding vector.
        """
        print("\nLoading embedding data. This may take a while.\n")
        print("embedding file: {}\nvoc limit: {}".format(
            embedding_file, voc_limit))
        try:
            # z = zipfile.ZipFile(embedding_file, "r")
            # with ZipFile(embedding_file, "r") as file_handle:
            fasttext_model = KeyedVectors.load_word2vec_format(
                embedding_file, binary=False, limit=voc_limit)

        except OSError:
            try:
                fasttext_model = KeyedVectors.load_word2vec_format(
                    embedding_file, binary=False, limit=voc_limit)

            except ValueError as e:
                print(
                    "\nValueError: {}\nPlease provide the correct data format "
                    "for the fasttext data. Abort.\n".format(e))
                sys.exit(1)
            # raise e
        print("\nEmbedding loaded.")
        print("Vocab size: {}, vector dims: {}".format(
            len(fasttext_model.wv.vocab), fasttext_model.vector_size))
        # return a word2vec dict and the embedding dimension
        return fasttext_model.wv, fasttext_model.vector_size

    def load_word2vec(self, embedding_file, voc_limit):
        """Prepare the embedding model from the given file.

        Returns:
            A dict mapping from a word to an embedding vector.
        """
        print("embedding file: {}\nvoc limit: {}".format(
            embedding_file, voc_limit))
        try:
            with gzip.open(embedding_file, "r") as file_handle:
                embedding_model = KeyedVectors.load_word2vec_format(
                    file_handle, binary=True, limit=voc_limit)

        except OSError:
            try:
                embedding_model = KeyedVectors.load_word2vec_format(
                    embedding_file, binary=True, limit=voc_limit)

            except ValueError as e:
                print(
                    "\nValueError: {}\nPlease provide the correct data format "
                    "for the word2vec data. Abort.\n".format(e))
                sys.exit(1)
            # raise e
        print("\nEmbedding loaded.")

        # return a word2vec dict and the embedding dimension
        return embedding_model.wv, embedding_model.vector_size

    def get_word_vectors(self):
        """Getter method to have access to the word vectors."""
        return self.word_vectors

    def get_minimal_distant_concept(self, word_vector, all_concepts):
        """Get the nearest concept.

        Args:
            word_vector:    numpy ndarray
                            The word vector.
            all_concepts:   list
                            A list of all concepts.
        Returns:
            The minimal distance and the associated concept.
        """
        if self.embedding_model_name == "word2vec":
            new_min_sim = self.word_vectors.distances(word_vector,
                                                      all_concepts)
            idx = np.argmin(new_min_sim)

            return new_min_sim[idx], all_concepts[idx]
        else:
            print("Not implemented for {}, abort.".format(
                self.embedding_model_name))
            sys.exit(1)

    def get_sim_btw_vector_and_concepts(self, word_vector, word_vec_name,
                                        all_concepts):
        """Get the similarity for each vector to all given concepts.

        Args:
            word_vector:    numpy ndarray
                            The word vector.
            word_vec_name:  str
                            The "word" or dummy.
            all_concepts:   list
                            A list of all concepts.
        Returns:
            A pandas dataframe comprising the instance and its distance
            to all concepts.
        """
        if self.embedding_model_name == "word2vec":
            # get the distances
            distances = self.word_vectors.distances(word_vector, all_concepts)
            # calculate the similarity score
            similarities = 1 - distances
            # create the dataframe with the concepts as columns
            df = pd.DataFrame(
                dict(zip(all_concepts, similarities)),
                columns=list(all_concepts),
                index=[word_vec_name])

            return df
        else:
            print("Not implemented for {}, abort.".format(
                self.embedding_model_name))
            sys.exit(1)

    def get_similarity_to_concepts(self, instance, concepts):
        """Get the nearest neighbors of an instance.

        Args:
            instance:   str
                        A word.
            concepts:   list
                        All available concepts
        Returns:
            An array of similarity scores for all concepts.
        """
        if self.embedding_model_name in ("word2vec", "glove", "fasttext"):
            sim_scores = []

            for concept in concepts:
                sim_scores.append(
                    self.word_vectors.similarity(instance, concept))
            sim_scores = np.asarray(sim_scores)

            return sim_scores
        else:
            sim_scores = []

            for concept in concepts:
                sim_scores.append(
                    self.word_vectors.similarity(instance, concept))
            sim_scores = np.asarray(sim_scores)

            print("Not implemented for {}, abort.".format(
                self.embedding_model_name))
            sys.exit(1)

    def get_embedding_for_word(self, word):
        """Get the embedding of one word.

        Args:
            word:   str
                    A word.

        Returns:
            The word vector for the given instance.
        """
        if self.embedding_model_name in ("word2vec", "glove", "fasttext"):
            if word in self.word_vectors:
                return self.word_vectors[word]
            return None
        else:
            tokens = self.tokenizer.tokenize(word)
            embeddings = np.array(
                [self.word_vectors[token] for token in tokens])

            return self.combine_word_vectors(embeddings)

    def combine_word_vectors(self, embeddings):
        """Combine two or more word vectors to one vector.

        Possible combination operations for embeddings include:
            max: np.ndarray.max(embeddings, axis=0)
            sum: np.ndarray.sum(embeddings, axis=0)
            average: np.ndarray.mean(embeddings, axis=0)

        Args:
            embeddings: numpy ndarray
                        A num_tokens x embedding_dim matrix with the
                        token embedding for each word of the compound
        Returns:
            combined_embedding: numpy ndarray
                                A combination of the given embeddings.
        """
        # the embeddings are combined according to the operation given in
        # the configuration
        combined = self.embedding_combiner(embeddings, axis=0)
        # make sure the resulting vector has the dimensions 1 x embedding_dim
        assert combined.shape == (self.embedding_dim, ), (
            "Please make sure that the shape of the combined vector is (1 x {}"
            "), currently it is {}.".format(self.embedding_dim,
                                            combined.shape))
        return combined

    def intersection_with_bert(self, word, max_length_of_instances):
        """..."""
        return self.in_bert(word, max_length_of_instances,
                            self.intersecting_embedding)

    def intersection_with_other_embedding(self, word, max_length_of_instances):
        """..."""
        return self.in_global_embedding(word, max_length_of_instances,
                                        self.intersecting_embedding)

    def in_bert(self, word, max_length_of_instances, lookup_dict):
        """Check if the given word is in the BERT vocabulary.

        For BERT, we don't have to modify the tokens, we just check if
        each word can be tokenized with the tokens given in the vocabulary.
        Compounds are later added / maxed / averaged (depends on the config).

        """
        all_in_vocab = True
        len_phrase = len(word.split(" "))

        if max_length_of_instances:
            if len_phrase > max_length_of_instances:
                return False

        tokens = self.tokenizer.tokenize(word)
        for token in tokens:
            if token in lookup_dict:
                continue
            else:
                all_in_vocab = False
                break
        if all_in_vocab:
            return word

        return False

    def in_global_embedding(self, word, max_length_of_instances, lookup_dict):
        """Check if a word is in the vocabulary of the given embedding."""
        if word in lookup_dict:
            return word

        if word.lower() in lookup_dict:
            return word.lower()

        word_list = word.split(" ")

        if max_length_of_instances:
            if len(word_list) > max_length_of_instances:
                return False

        # try two different variants of modification
        # e.g. "new york" --> "new_york"
        variant_1 = word.replace(" ", "_")

        if variant_1 in lookup_dict:
            return variant_1

        # e.g. "new york" --> "New_York"
        variant_2 = "_".join(
            [x[0].upper() + x[1:] for x in word_list if x != ""])

        if variant_2 in lookup_dict:
            return variant_2

        # e.g. "new York" --> "new_york"
        variant_3 = word.lower().replace(" ", "_")

        if variant_3 in lookup_dict:
            return variant_3
        # create all possible phrases from the given word list in the
        # given order
        # the longest phrases in the vocabulary will be returned
        if len(word_list) > 1:
            longest_phrase = ""
            for i in range(1, len(word_list)):
                new_phrase = "_".join(word_list[0:i + 1])
                if new_phrase in lookup_dict:
                    longest_phrase = new_phrase

            if longest_phrase != "":
                return longest_phrase

        return False

    def in_embedding(self, word, max_length_of_instances=5):
        """Check if word or phrase or its modified version is in embedding.

        Args:
            word:   str
                    A word.
            max_length_of_instances:   bool/int
                                        Consider only phrases of this
                                        length in the data.
        Returns:
            An instance (str), if the instance or a modified version are in the
            embedding vocabulary.
            False, if not.
        """
        if self.embedding_model_name in ("word2vec", "glove", "fasttext"):

            in_global = self.in_global_embedding(word, max_length_of_instances,
                                                 self.word_vectors)

            if not in_global:
                return False
            # check if the word can be found in Bert's vocab as well
            if self.check_other_embedding == "bert":
                # if yes, return the modified word
                if self.intersection_with_bert(word, max_length_of_instances):
                    return in_global

                # if no, return nothing
                return False

            # if the other-embedding-check is not required, continue
            # with the word2vec version of the given word
            return in_global

        # assume the other possible embedding is Bert
        else:
            in_bert = self.in_bert(word, max_length_of_instances,
                                   self.word_vectors)

            if not in_bert:
                return False
            if self.check_other_embedding == "word2vec":
                # if yes, return the modified word
                if self.intersection_with_other_embedding(
                        word, max_length_of_instances):
                    return in_bert

                # if no, return nothing
                return False

            # if the other-embedding-check is not required, continue
            # with the bert version of the given word
            return in_bert

    def convert_examples_to_features(self, examples, seq_length):
        """Load a data file into a list of `InputFeature`s.

        Taken from https://github.com/huggingface/pytorch-pretrained-BERT/blob/
                   master/examples/extract_features.py

        Args:
            examples:   list
                        A list of InputExample objects.
            seq_length: int
                        The maximal sequence length.
        Returns:
            A list of InputFeature objects.
        """
        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = self.tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = self.tokenizer.tokenize(example.text_b)

            if tokens_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                utils._truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > seq_length - 2:
                    tokens_a = tokens_a[0:(seq_length - 2)]
            """
            The convention in BERT is:
            (a) For sequence pairs:
             tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not .
                       [SEP]
             type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1
                         1
            (b) For single sequences:
             tokens:   [CLS] the dog is hairy . [SEP]
             type_ids:   0   0   0   0  0     0   0

            Where "type_ids" are used to indicate whether this is the first
            sequence or the second sequence. The embedding vectors for `type=0`
             and
            `type=1` were learned during pre-training and are added to the
            wordpiece
            embedding vector (and position vector). This is not *strictly*
            necessary
            since the [SEP] token unambigiously separates the sequences, but it
            makes
            it easier for the model to learn the concept of sequences.

            For classification tasks, the first vector (corresponding to [CLS])
            is
            used as as the "sentence vector". Note that this only makes sense
            because
            the entire model is fine-tuned.
            """
            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            if tokens_b:
                for token in tokens_b:
                    tokens.append(token)
                    input_type_ids.append(1)
                tokens.append("[SEP]")
                input_type_ids.append(1)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens.
            # Only real tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            assert len(input_type_ids) == seq_length

            if ex_index == 0:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (example.unique_id))
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info(
                    "input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("input_type_ids: %s" % " ".join(
                    [str(x) for x in input_type_ids]))

            features.append(
                BertInputFeatures(
                    unique_id=example.unique_id,
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_type_ids=input_type_ids))
        return features


class BertInputExample(object):
    """The input to Bert."""

    def __init__(self, unique_id, text_a, text_b):
        """..."""
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class BertInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask,
                 input_type_ids):
        """..."""
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
