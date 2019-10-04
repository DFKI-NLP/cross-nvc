##!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utils for visualizing the models' outcomes.

@author lisa.raithel@dfki.de
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Visualizer(object):
    """A visualizer for concept activation profiles etc."""

    def __init__(self,
                 model_path="",
                 details_str="",
                 pretrained=False,
                 debug=False):
        """Initialize the NVC model class.

        Args:
            model_path:    str
                           The path to the model directory.
            details_str:    str
                            A str with all necessary information to build
                            unique file names.
            pretrained: bool
                        Using a pre-trained model or not.
        """
        self.model_path = model_path
        self.details_str = details_str
        self.pretrained = pretrained
        self.debug = debug

    def update_paths(self, model_path="", details_str="", pretrained=False):
        """Update the paths."""
        self.model_path = model_path
        self.details_str = details_str
        self.pretrained = pretrained

    def write_to_csv(self, dataframe, file_name, index=False):
        """Write a dataframe to a CSV file.

        Args:
            dataframe:  pandas dataframe
                        A dataframe with activations.
            file_name:  str
                        The file name of the resulting CSV.
            index:  bool
                    If true, write row names.
        """
        if not self.debug:
            index_label = False
            if index:
                index_label = "instance"

            dataframe.to_csv(
                path_or_buf=file_name,
                sep=" ",
                header=True,
                index_label=index_label,
                index=index)

    def transpose_df_and_write_to_file(self, df, file_name):
        """Transpose the dataframe for better plotting (from CSV file).

        Args:
            df: pandas dataframe
                The data.
            file_name:  str
                        The file name for the CSV file.
        """
        # transpose the data and add a new index
        df_transposed = df.transpose()
        df_transposed = df_transposed.reset_index()
        df_transposed = df_transposed.rename(columns={
            "index": "concept",
            0: "activation"
        })
        if not self.debug:
            # write transposed dataframe to CSV file
            df_transposed.to_csv(
                path_or_buf=file_name,
                sep=";",
                header=True,
                index_label="number",
                index=True)

    def get_top_n_concepts(self, predictions, concepts, top_n=3):
        """..."""
        # get the indices of the number of concepts that are to be highlighted.
        max_idx = predictions.argsort()[-top_n:][::-1]
        # print(max_idx)
        # sort the indices to match the order of concepts
        # max_idx.sort()
        # print(max_idx)
        # get the actual scores
        # max_activ = predictions[max_idx]

        # get the actual concepts
        max_concepts = [concepts[i] for i in max_idx]
        return max_concepts

    def get_n_largest_labels(self, predictions, classes, max_highlights=3):
        """Return a list of the column names of the n largest values in the df.

        Args:
            predictions:    numpy ndarray
                            The output of the model for a certain instance.
            classes:        list
                            All available concepts.
            max_highlights: int
                            The number of concepts to be returned.

        Returns:
            A list of the maximal activated concepts.
        """
        # get the indices of the number of concepts that are to be highlighted.
        max_idx = predictions.argsort()[-max_highlights:][::-1]
        # print(max_idx)
        # sort the indices to match the order of concepts
        max_idx.sort()
        # print(max_idx)
        # get the actual scores
        # max_activ = predictions[max_idx]

        # get the actual concepts
        max_concepts = [classes[i] for i in max_idx]
        return max_concepts

    def write_activations_to_csv(self, dataframe, instance, num_col, language):
        """Write the activations to a CSV file.

        Args:
            dataframe:  pandas dataframe
                        The dataframe with the activations.
            instance:   str
                        The instance whose activations are plotted.
            num_col:    int
                        The column number to decide on the file name.
        """
        # the left column (0) of the plot is concept activations,
        # the right column (1) is cosine similarities
        if num_col == 0:
            s = "concept_activations"
        else:
            s = "cosine_sim"

        # if we use a pre-trained model, mark this in the file name
        if self.pretrained:
            # transpose the data for better format for plotting from CSV
            self.transpose_df_and_write_to_file(
                dataframe, "{}/{}_{}_transposed_pretrained_{}_{}.csv".format(
                    self.model_path, instance, s, self.details_str, language))

        else:
            self.transpose_df_and_write_to_file(
                dataframe, "{}/{}_{}_transposed_{}_{}.csv".format(
                    self.model_path, instance, s, self.details_str, language))

    def plot_activation_profile(self,
                                classes=[],
                                values=[],
                                instance="",
                                num_row=0,
                                num_col=0,
                                y_label="",
                                x_label="",
                                max_highlights=3,
                                axes=None,
                                num_instances=2,
                                language="english"):
        """Plot a dataframe and annotate the highest bars.

        Args:
            classes:    list
                        A list of concepts.
            values: list
                    A list of activations.
            instance:   str
                        The instance that is plotted.
            num_row:    int
                        The plot row number for the current instance.
            num_col:    int
                        The plot column number (0 or 1 for activations
                        or similarity).
            y_label:    str
                        The label of the y-axis.
            x_label:    str
                        The label of the x-axis.
            max_highlights: int
                            The number of concepts that should be labeled.
            axes:   numpy ndarray
                    The axes of the plot.
            num_instances:  int
                            The number of instances and with that, rows.
        """
        # create a dataframe for each plot
        df = pd.DataFrame(data=dict(zip(classes, values[num_row])), index=[0])
        # get the concepts for the highest activations
        max_concepts = self.get_n_largest_labels(
            values[num_row], classes, max_highlights=max_highlights)

        concepts_selected = []
        for c in classes:
            if c in max_concepts:
                concepts_selected.append(c)
            else:
                concepts_selected.append("")

        df_2 = pd.DataFrame(
            data=dict(zip(classes, concepts_selected)),
            index=["concepts_selected"])
        df_3 = df.append(df_2)
        # write activations to a csv
        self.write_activations_to_csv(
            dataframe=df_3,
            instance=instance,
            num_col=num_col,
            language=language)

        # Depending on how many instances are to be plotted, specify
        # the position in the "grid".
        # Edge case for single vector plotting (without cosine) for one
        # instance only --> "'AxesSubplot' object does not support indexing"
        # => hack: for single vectors, a second dummy plot is created to
        # keep things simple
        if num_col is False:
            # do not use "if num_col".
            # We only want num_col = False, not num_col = 0
            position = num_row

        elif num_instances == 1:
            position = num_col

        else:
            position = num_row, num_col

        axes[position].set_xlabel(x_label)
        axes[position].set_ylabel("{} of instance '{}'".format(
            y_label, instance))

        axes[position].tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)

        # plot the activations
        sns.barplot(
            data=df, color="black", ax=axes[position])  # palette="deep"

        # annotate the (highest) bars with their concepts
        for j, _class in enumerate(classes):
            if _class in max_concepts:
                axes[position].annotate(
                    classes[j],
                    xy=(j, df.values[0][j]),
                    fontsize=8,
                    xytext=(0, 20),
                    textcoords="offset points",
                    rotation=50)

    def display_contextualized_concept_activations(self,
                                                   sentence="",
                                                   word="",
                                                   classes=[],
                                                   predictions={},
                                                   max_highlights=3):
        """..."""
        num_occurrences = len(predictions) - 1
        print("sentence: {}".format(sentence))

        # get the prediction for the embedding without context
        global_prediction = predictions.pop("global", None)

        # clear old plots
        plt.clf()
        # create subplots to set activation and similarity profiles next
        # to each other
        fig, axes = plt.subplots(
            nrows=num_occurrences, ncols=2, figsize=(20, 20), dpi=200)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)

        # plot the activations and similarities for every instance
        for position, pred in predictions.items():

            self.plot_activation_profile(
                classes=classes,
                values=global_prediction,
                num_row=position,
                num_col=0,
                instance=word,
                y_label="activation",
                x_label="concepts",
                max_highlights=max_highlights,
                axes=axes,
                num_instances=num_occurrences)

            self.plot_activation_profile(
                classes=classes,
                values=pred,
                num_row=position,
                num_col=1,
                instance=word,
                y_label="activation",
                x_label="concepts",
                max_highlights=max_highlights,
                axes=axes,
                num_instances=num_occurrences)

            if self.pretrained:
                eps_name = "{}/{}_{}_pretrained.eps".format(
                    self.model_path, "_".join(word), self.details_str)
            else:
                eps_name = "{}/{}_{}.eps".format(
                    self.model_path, "_".join(word), self.details_str)
            # remove the axis borders
            sns.despine(bottom=True)
            if not self.debug:
                plt.savefig(eps_name, dpi=300)

        plt.show()

    def display_concept_activations_without_cosine(self,
                                                   instances=[],
                                                   classes=[],
                                                   predictions=[],
                                                   max_highlights=3,
                                                   language="english"):
        """Plot activation profile of one instance."""
        num_instances = len(instances)
        # add dummy plot to keep things simple:
        # 'AxesSubplot' object does not support indexing when plotting a
        # subplot of 1x1
        if num_instances == 1:
            num_instances = 2
        # clear old plots
        plt.clf()

        fig, axes = plt.subplots(
            nrows=num_instances, ncols=1, figsize=(20, 20), dpi=200)

        for i, instance in enumerate(instances):
            self.plot_activation_profile(
                classes=classes,
                values=predictions,
                num_row=i,
                num_col=False,
                instance=instance,
                y_label="activation",
                x_label="concepts",
                max_highlights=max_highlights,
                axes=axes,
                num_instances=num_instances,
                language=language)

            if self.pretrained:
                eps_name = "{}/{}_{}_{}_pretrained.eps".format(
                    self.model_path, "_".join(instances), self.details_str,
                    language)
            else:
                eps_name = "{}/{}_{}_{}.eps".format(self.model_path,
                                                    "_".join(instances),
                                                    self.details_str, language)
            # remove the axis borders
            sns.despine(bottom=True)
            if not self.debug:
                plt.savefig(eps_name, dpi=300)

        plt.show()

    def display_concept_activations(self,
                                    instances=[],
                                    classes=[],
                                    predictions=[],
                                    similarities=[],
                                    max_highlights=3):
        """Plot the activation profiles.

        Args:
            instances:  list
                        A list of words whose activation profiles are to be
                        plotted.
            classes:    list
                        A list of concepts.
            predictions:    numpy ndarray
                            The predictions of the model.
            similarities:   list
                            A list of cosine similarities.
            max_highlights: int
                            The number of concepts that should be labeled.
            model_path: str
                        The path to the model (directory).
        """
        num_instances = len(instances)

        # clear old plots
        plt.clf()
        # create subplots to set activation and similarity profiles next
        # to each other
        fig, axes = plt.subplots(
            nrows=num_instances, ncols=2, figsize=(20, 20), dpi=200)

        plt.subplots_adjust(hspace=0.5, wspace=0.5)

        # plot the activations and similarities for every instance
        for i, instance in enumerate(instances):

            self.plot_activation_profile(
                classes=classes,
                values=predictions,
                num_row=i,
                num_col=0,
                instance=instance,
                y_label="activation",
                x_label="concepts",
                max_highlights=max_highlights,
                axes=axes,
                num_instances=num_instances)

            self.plot_activation_profile(
                classes=classes,
                values=similarities,
                num_row=i,
                num_col=1,
                instance=instance,
                y_label="similarity",
                x_label="concepts",
                max_highlights=max_highlights,
                axes=axes,
                num_instances=num_instances)

            if self.pretrained:
                eps_name = "{}/{}_{}_pretrained.eps".format(
                    self.model_path, "_".join(instances), self.details_str)
            else:
                eps_name = "{}/{}_{}.eps".format(
                    self.model_path, "_".join(instances), self.details_str)
            # remove the axis borders
            sns.despine(bottom=True)
            if not self.debug:
                plt.savefig(eps_name, dpi=300)

        plt.show()

    def display_way(self,
                    vec_names,
                    values,
                    annotation,
                    words,
                    steps_between,
                    x_label="instance",
                    y_label="activation"):
        """Plot a list of vectors.

        Args:
            vec_names:  list
                        A list of 'vector names'.
            values: numpy ndarray
                    The vectors.
            annotation: list
                        A list of annotations (concepts) for the vector
                        activations.
            words:  list
                    A list containing the start and end vector name.
            steps_between:  int
                            The number of vectors that were sampled.
            x_label:    str
                        The label for the x-axis.
            y_label:    str
                        The label for the y-axis.
        """
        df = pd.DataFrame({
            x_label: vec_names,
            y_label: values,
            'concept': annotation
        })

        if self.pretrained:
            self.write_to_csv(
                df, "{}/max_activations_sampling_{}_pretrained_{}_{}_k{}_"
                "{}.csv".format(self.model_path, y_label, words[0], words[1],
                                steps_between, self.details_str))

        else:
            self.write_to_csv(
                df, "{}/max_activations_sampling_{}_{}_{}_k{}_{}.csv".format(
                    self.model_path, y_label, words[0], words[1],
                    steps_between, self.details_str))

        g = sns.FacetGrid(df, height=7)

        def plotter(x, y, **kwargs):
            regplot = sns.regplot(
                data=df,
                x=x_label,
                y=y_label,
                fit_reg=False,
                marker="x",
                color="darkred")

            plt.plot(x, y, linewidth=1, color="darkred")

            tick_labels = regplot.get_xticklabels()

            for j, tick in enumerate(tick_labels):
                # rotate all labels by 90 degrees
                tick.set_rotation(90)
                tick.set_weight("light")
                if j == 0 or j == len(tick_labels) - 1:
                    tick.set_weight("normal")

            for i in range(len(x)):
                plt.annotate(
                    annotation[i],
                    xy=(i, y.values[i]),
                    fontsize=8,
                    xytext=(0, 50),
                    textcoords="offset points",
                    rotation=90)

        g.map(plotter, x_label, y_label)

        file_name = "{}/vector_way_{}_{}_{}_{}.eps".format(
            self.model_path, y_label, words[0], words[1], self.details_str)
        plt.savefig(file_name)

        plt.show()

    def show_samples(self, activations, concepts, ticks_text, first_word,
                     last_word, steps_between):
        """Show the 'way of activations' from one vector to another.

        Args:
            activations:    pandas dataframe
                            The activation profile.
            similarities:   pandas dataframe
                            The similarity profile.
            concepts:       list
                            All available concepts.
            ticks_text:     list of str
                            The labels for the graph ticks.
            first_word:     str
                            The first word of the sampling process.
            last_word:      str
                            The last word of the sampling process.
            steps_between:  int
                            The number of sampling steps.
            model_path:     str
                            The path to the model directory.
        """
        max_concepts_nvc = []
        max_activations = []

        self.write_to_csv(
            activations, "{}/sampling_{}_{}_k{}_{}.csv".format(
                self.model_path, first_word, last_word, steps_between,
                self.details_str))

        # get the maximal activated concepts and their activations for
        # each instance
        for instance, activ_series in activations.iterrows():
            max_concept = activ_series.idxmax()
            max_activation = activ_series[max_concept]

            max_concepts_nvc.append(max_concept)
            max_activations.append(max_activation)

        # max_concepts_cosine = []
        # max_similarities = []

        # get the maximal similar concepts and their similarity for
        # each instance
        # for instance, sim_series in similarities.iterrows():
        #     max_concept = sim_series.idxmax()
        #     max_sim = sim_series[max_concept]

        #     max_concepts_cosine.append(max_concept)
        #     max_similarities.append(max_sim)

        self.display_way(
            vec_names=ticks_text,
            values=max_activations,
            annotation=max_concepts_nvc,
            words=[first_word, last_word],
            y_label="activation",
            steps_between=steps_between)

        # self.display_way(
        #     vec_names=ticks_text,
        #     values=max_similarities,
        #     annotation=max_concepts_cosine,
        #     words=[first_word, last_word],
        #     y_label="similarity",
        #     steps_between=steps_between)


if __name__ == '__main__':
    import numpy as np
    visualizer = Visualizer(
        model_path="test/", details_str="test_plots", debug=False)

    predictions = [
        np.array([0.5, 0.1, 0.3, 0.02, 0.004]),
        np.array([0.1, 0.5, 0.6, 0.3, 0.04])
    ]

    similarities = [
        np.array([0.3, 0.1, 0.4, 0.2, 0.6]),
        np.array([0.2, 0.5, 0.3, 0.6, 0.9])
    ]
    visualizer.display_concept_activations_without_cosine(
        instances=["beagle"],  # ,"puma"],
        classes=["dog", "cat", "predator", "bear", "fish"],
        predictions=predictions,
        max_highlights=2)

    # for prediction in predictions:
    #     print("shape prediction: {}".format(prediction.shape))

    #     top = visualizer.get_top_n_concepts(
    #         prediction,
    #         concepts=["dog", "cat", "predator", "bear", "fish"],
    #         top_n=4)

    #     print(top)

    # wanted: [dog, predator, cat]
    #         [predator, cat, dog]
    # visualizer.display_concept_activations(
    #     instances=["beagle", "puma"],
    #     classes=["dog", "cat", "predator", "bear", "fish"],
    #     predictions=predictions,
    #     similarities=similarities,
    #     max_highlights=3)
