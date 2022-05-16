# Crosslingual Neural Vector Conceptualization (NVC)

Applying NVC to aligned word vectors of English and Chinese.


Accompanying code for the paper:
```
@inproceedings{raithel_cross_nvc_2019,
  title = {Crosslingual Neural Vector Conceptualization},
  booktitle = {NLPCC Workshop on Explainable Artificial Intelligence},
  author = {Raithel, Lisa and Schwarzenberg, Robert},
  year = {2019}
  }
```
## Installation

Create and activate an environment with Python 3.6.

```
conda create --name CNVC python=3.6
source activate CNVC
```

After cloning the repository, install the requirements.

```
pip install -r requirements.txt
```



## Data

### Word Vectors:

The current underlying word vectors were learned with the fastText model [2]. Please download the pre-trained word vectors for English and Chinese (or any other second language for that matter) to the `data` directory: 

```
wget -P ~/cross_nvc/src/data/ "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec"
wget -P ~/cross_nvc/src/data/ "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.zh.align.vec"
```
### Microsoft Concept Graph

For using the Microsoft Concept Graph (MCG) data from scratch, please download the data here: 

```
wget -P ~/cross_nvc/src/data/ "https://concept.research.microsoft.com/Home/StartDownload"
```
The MCG data dump does only comprise concepts, instances and associated counts, no probabilities or REP values. These need to be calculated before training. Therefore, see the script `utils/ms_concept_graph_scoring.py` in the notebook which calculates all needed probabilities and writes the data to a TSV file.

This takes some time, therefore we recommend to download the already preprocessed data from [here](https://drive.google.com/drive/folders/1ag-6Rsj3oAT7LSVznvxowFGR3lmDuhLI) and unzip into
```
./src/data/
```
This dump includes
  - the data from the MCG as TSV file with REP values (`data-concept-instance-relations-with-rep.tsv`)
  - a JSON file of the same data, but filtered for instances that also occur in the fastText vocabulary (`raw_data_dict_fasttext_maxlenFalse.json`)
  - the filtered data including only concepts that have at least 100 instance with a REP > -10 (`filtered_data_i100_v-10_fasttext_maxlenFalse`)
  - Chinese translations of the English test dataset (`translations_aligned`)
  - the test set on which we reported the scores in the paper (`test_set.csv`)

**We recommend to use the preprocessed data `input_data/raw_data_dict_fasttext_maxlenFalse.json` which includes all concepts and instances with their corresponding REP values in a JSON file.**

## Run and Replicate Experiments with the Notebook

The jupyter notebook `demo_cross_nvc.ipynb` demonstrates how to use our (pre-)trained neural vector conceptualization (NVC) model to display the reported activation profiles.

Navigate to `src/` and start the notebook:

```
jupyter notebook demo_cross_nvc.ipynb 
```

or 

```
export CUDA_VISIBLE_DEVICES=DEVICE_NUM jupyter notebook demo_cross_nvc.ipynb
```
if you want to run it on GPU DEVICE_NUM.


You can run two versions of the notebook:
  1. Use our pre-trained NVC model
  2. Train a new model

### Use the pre-trained model

If you run all cells of the notebook in the given order and without changing anything except the necessary file paths, our pre-trained cross-NVC model is applied to a given filtered dataset and the results are reported at the end of the notebook.


### Train a new model

Follow the instructions in the notebook to train and evaluate a new model.


[1] A. Joulin, P. Bojanowski, T. Mikolov, H. Jegou, E. Grave, Loss in Translation: Learning Bilingual Word Mapping with a Retrieval Criterion
[2] P. Bojanowski, E. Grave, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information
