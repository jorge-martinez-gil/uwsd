
# Context-Aware Semantic Similarity Measurement for Unsupervised Word Sense Disambiguation

This repository houses the codebase for replicating the experiments detailed in Jorge Martinez-Gil's paper on Context-Aware Semantic Similarity Measurement for Unsupervised Word Sense Disambiguation. Discover more insights and applications through our [arXiv preprint](https://arxiv.org/abs/2305.03520) and an accessible [Medium article](https://medium.com/@jorgemarcc/applications-of-context-aware-semantic-similarity-9c62492be392).

[![arXiv preprint](https://img.shields.io/badge/arXiv-2305.03520-brightgreen.svg)](https://arxiv.org/abs/2305.03520) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Citations](https://img.shields.io/badge/citations-3-blue)](https://scholar.google.com/citations?view_op=view_citation&hl=en&citation_for_view=X1pRUYcAAAAJ:7XUxBq3GufIC)

![Summary](uwsd.png)

## 🌍 Overview 

Word sense disambiguation (WSD) plays an important role in Natural Language Processing (NLP). It involves deciphering the intended meaning of a word in a multi-sense context, which is crucial for improving the performance of applications like machine translation and information retrieval.

Our repository offers an innovative unsupervised approach to WSD using context-aware semantic similarity:

1. **Preprocessing**: Clean and prepare the text data.
2. **Context Extraction**: Identify the context surrounding the ambiguous word.
3. **Semantic Similarity**: Utilize pre-trained sentence embeddings and cosine similarity to evaluate semantic parallels.
4. **Sense Selection**: Choose the sense with the highest similarity score.

Included are the necessary code, pre-trained embeddings, and test data for thorough evaluation.

## 🛠️ Installation 

```bash
pip install -r requirements.txt
```

## 📊 Dataset 

The CoarseWSD-20 dataset, a well-known resource for coarse-grained WSD, forms the backbone of our experiments. It includes 20 commonly ambiguous words.

## 🚀 Usage Guide 

Follow these steps to apply our method:

1. Clone this repository.
2. Install dependencies (refer to the installation section).
3. Download and position pre-trained word embeddings in the data directory.
4. Execute the script of your choice and observe the results in your console.
5. For best performance, use scripts prepared to work with GPUs.

## 📝 Evaluation 

Evaluate our approach using the provided test data.

**Unsupervised Word Sense Disambiguation (UWSD):**
 - `python uwsd_bert.py` - BERT
 - `python uwsd_elmo.py` - ELMo
 - `python uwsd_use.py` - Universal Sentence Encoder (USE)
 - `python uwsd_wmd.py` - Word Mover's Distance (WMD)

**Context-Aware Semantic Similarity (CASS):**
 - `python cass-wordnet+bert.py` - CASS using WordNet and BERT
 - `python cass-word2vec+bert.py` - CASS using word2vec and BERT
 - `python cass-webscrapping+bert.py` - CASS using webscraping and BERT

**Example Scenario UWSD:**
 - *Typed object-oriented programming languages, such as **java** and c++ , often do not support first-class methods*
--> *options* (island, programming language)
	 - uwsd_bert: **programming language** 
	 - uwsd_elmo: **programming language**
	 - uwsd_use: **programming language** 
	 - uwsd_wmd: **programming language** 
	 - ChatGPT-4: **programming language**

**Example Scenario CASS:**
- *Vienna is a nice city situated in the **center** of the European continent.*
  - cass-wordnet+bert: **middle**
  - cass-word2vec+bert: **hub**
  - cass-webscrapping+bert: **mid**
  - ChatGPT-4: **middle**

## 📈 Performance Results 

The summary of the results in terms of the CoarseWSD-20 dataset disambiguation is:

| Strategy  |  Hits  |  Accuracy |
| ------------ | ------------ | ------------ |
| UWSD+BERT  |  7,927  | 77.74%   |
| MFS-Baseline  | 7,487  |  73.43% |
| UWSD+USE | 7,335 | 71.94% |
| UWSD+ELMo | 7,010 | 68.75% |
| UWSD+WMD | 5,868 | 57.55% |
| RO-Baseline | 4,459 | 43.73% |

## 📚 Citation 

If you utilize our work, kindly cite us:

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

## 📖 Research that has cited this work

1. **[Pantip Multi-turn Datasets Generating from Thai Large Social Platform Forum Using Sentence Similarity Techniques](https://ieeexplore.ieee.org/iel8/10799229/10799211/10799403.pdf)**
   - **Authors:** A. Sae-Oueng, K. Kerdthaisong, …
   - **Conference:** *Joint Symposium on …*, 2024 (IEEE)
   - **Abstract:** Fine-tuning Large Language Models (LLMs) for specific domains is crucial. However, the lack of Thai open dialogues presents a major challenge. For this challenge, the study proposes generating multi-turn dialogue datasets from Pantip, a large Thai social platform forum, using sentence similarity techniques.
  
2. **[Assessing GPT’s Potential for Word Sense Disambiguation: A Quantitative Evaluation on Prompt Engineering Techniques](https://doi.org/10.1109/icsgrc62081.2024.10691283)**
   - **Authors:** D. Sumanathilaka, N. Micallef, J. Hough
   - **Conference:** *IEEE 15th Control and System Graduate Research Colloquium (ICSGRC)*, 2024
   - **Abstract:** Tests few-shot, chain-of-thought, and direct prompts on GPT-3.5 and GPT-4 for word sense disambiguation in social media text. Chain-of-thought with GPT-4 scored highest without extra tuning.
  
3. **[GlossGPT: GPT for Word Sense Disambiguation using Few-shot Chain-of-Thought Prompting](https://www.sciencedirect.com/science/article/pii/S1877050925008385/pdf?md5=293d431d6e660d73e125df97da5d2804&pid=1-s2.0-S1877050925008385-main.pdf)**  
   - **Authors:** D. Sumanathilaka, N. Micallef, J. Hough  
   - **Conference:** *Procedia Computer Science*, 2025 (Elsevier)  
   - **Abstract:** Lexical ambiguity is a major challenge in computational linguistic tasks, as limitations in proper sense identification lead to inefficient translation and question answering. This study presents *GlossGPT*, a GPT-based few-shot method for word sense disambiguation using chain-of-thought prompting. Incorporating gloss definitions in the prompt improves sense resolution across multiple benchmark datasets.


## 📄 License 

Released under the MIT License. [View License](LICENSE).
