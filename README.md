# Context-Aware Semantic Similarity Measurement for Unsupervised Word Sense Disambiguation
This repository contains code for reproducing the experiments reported in the paper
- Jorge Martinez-Gil, "Context-Aware Semantic Similarity Measurement for Unsupervised Word Sense Disambiguation", [[arXiv preprint]](https://arxiv.org/abs/2305.03520), May 2023

There is also available is an article on Medium intended for a general audience.
- [[Applications of Context-Aware Semantic Similarity]](https://medium.com/@jorgemarcc/applications-of-context-aware-semantic-similarity-9c62492be392)

# Overview
Word sense disambiguation is the task of determining the meaning of a word in context, where a word can have multiple meanings or senses. This task is important in NLP, as it can help improve the accuracy of various downstream applications, such as machine translation and information retrieval.

This repository provides an approach for unsupervised word sense disambiguation using context-aware semantic similarity measurement. The approach involves the following steps:

- Preprocess the input text.
- Extract the context of the ambiguous word, including a window of surrounding words.
- Calculate the semantic similarity between the context of the ambiguous word and the context of each possible sense of the word, using pre-trained sentence embeddings and a measure of cosine similarity.
- Select the sense with the highest semantic similarity score as the correct sense of the ambiguous word.

The repository includes code for implementing this approach, as well as pre-trained word embeddings and test data for evaluation.

# Install
``` pip install -r requirements.txt```

# Dataset
We use an adaption of the CoarseWSD-20 dataset. CoarseWSD-20 is a publicly available dataset for coarse-grained word sense disambiguation, containing instances of 20 ambiguous words.

# Usage
To use this approach for unsupervised word sense disambiguation, follow these steps:

- Clone the repository to your local machine.
- Install the required libraries and tools (see above).
- Download and extract the pre-trained word embeddings to the data directory.
- Run the corresponding script (see below)
- The script will output the results to the console.

# Evaluation
The repository includes test data for evaluating the performance of the context-aware semantic similarity measurement approach. To run the evaluation, follow these steps:

``` python uwsd_bert.py```
Run the UWSD program using BERT sentence embeddings.

``` python uwsd_elmo.py```
Run the WSD program using ELMo sentence embeddings.

``` python uwsd_use.py```
Run the UWSD program using Universal Sentence Encoder (USE) embeddings.

``` python uwsd_wmd.py```
Run the UWSD program using Words Mover Distance (WMD) approach.

Additionally, a program is included that allows to conveniently calculate context-aware semantic similarity for inclusion in software projects.

``` python cass-wordnet+bert.py```
Run a CASS program using WordNet for synonyms candidates and the best possible approach for disambiguating (i.e., BERT sentence embeddings)

``` python cass-word2vec+bert.py```
Run a CASS program using word2vec for synonyms candidates and the best possible approach for disambiguating (i.e., BERT sentence embeddings)

``` python cass-webscrapping+bert.py```
Run a CASS program using webscrapping for synonyms candidates and the best possible approach for disambiguating (i.e., BERT sentence embeddings)

Example: Calculate a word being semantically equivalent to **center** in the sentence *Vienna is a nice city situated in the center of the european continent*.	

- cass-wordnet+bert: **middle**
- cass-word2vec+bert: **hub**
- cass-webscrapping+bert: **mid**
- ChatGPT-4: **middle**


## Results
The summary of the results in terms of the CoarseWSD-20 dataset disambiguation is:

| Strategy  |  Hits  |  Accuracy |
| ------------ | ------------ | ------------ |
| UWSD+BERT  |  7,927  | 77.74%   |
| MFS-Baseline  | 7,487  |  73.43% |
| UWSD+ELMo | 7,010 | 68.75% |
| UWSD+USE | 6,396 | 62.73% |
| UWSD+WMD | 5,868 | 57.55% |
| RO-Baseline | 4,459 | 43.73% |


## Citation
If you use this work, please cite:

```
@inproceedings{martinez2023b,
  author    = {Jorge Martinez-Gil},
  title     = {Context-Aware Semantic Similarity Measurement for Unsupervised Word Sense Disambiguation},
  journal   = {CoRR},
  volume    = {abs/2305.03520},
  year      = {2023},
  url       = {https://arxiv.org/abs/2305.03520},
  doi       = {https://doi.org/10.48550/arXiv.2305.03520},
  eprinttype = {arXiv},
  eprint    = {2305.03520}
}

```

# License
This code is released under the MIT License. See the LICENSE file for more information.