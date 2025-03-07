# -*- coding: utf-8 -*-
"""
[Martinez-Gil2023b] Context-Aware Semantic Similarity Measurement for Unsupervised Word Sense Disambiguation, arXiv preprint arXiv:2305.03520, 2023

@author: Jorge Martinez-Gil
"""

# Install required packages (Colab)
!pip install tensorflow tensorflow-hub

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import json
from scipy.spatial import distance

# Check and display available GPU device.
device_name = tf.test.gpu_device_name()
print("GPU device:" if device_name else "No GPU device found.", device_name)

# Load ELMo from TensorFlow Hub.
# The TF Hub model automatically uses GPU when available.
elmo = hub.load("https://tfhub.dev/google/elmo/3")
print("ELMo model loaded from TF Hub.")

def get_sentence_embeddings(sentences):
    """
    Given a list of sentences (strings), returns a NumPy array with 
    one sentence-level embedding per sentence (by averaging over token embeddings).
    """
    # Convert list of sentences to a tensor.
    sentences_tensor = tf.constant(sentences)
    # Call the default signature to get embeddings.
    embeddings = elmo.signatures["default"](sentences_tensor)["elmo"]
    # Average the embeddings over the tokens (axis=1) to get a single vector per sentence.
    sentence_embeddings = tf.reduce_mean(embeddings, axis=1)
    return sentence_embeddings.numpy()

def find_token_position(token, lst):
    """
    Returns the index of the first element in lst that contains the token.
    """
    for i, s in enumerate(lst):
        if token in s:
            return i
    return 0

def accumulate(synonyms, word, context):
    """
    For each candidate synonym, create a modified sentence.
    Appends the source sentence (if not already added) and the candidate sentence.
    """
    global sentences  # Use the global sentence accumulator.
    for i in range(len(synonyms)):
        cons = synonyms[i]
        cons = cons.replace('_', ' ').replace('(', '').replace(')', '')
        tokens_to_check = cons.split()
        for token in tokens_to_check:
            if token.lower() == word.lower():
                cons = cons.replace(token, "").replace(' ', '')
        if word.lower() != cons.lower():
            target = context.replace(word, cons)
            source = context
            if source not in sentences:
                sentences.append(source)
            sentences.append(target)
    return 1

# Containers to accumulate overall gold labels and predictions.
overall_gold = []
overall_pred = []

# Define the path to your data directory.
parent_dir = '/content/drive/MyDrive/datasets/CoarseWSD-20'

# Global container for sentences.
sentences = []

# Process each folder in the parent directory.
for folder in os.listdir(parent_dir):
    synonyms = []
    nums = []
    data = []
    results = []
    
    folder_path = os.path.join(parent_dir, folder)
    
    # Load synonyms from classes_map.txt.
    file_path = os.path.join(folder_path, 'classes_map.txt')
    with open(file_path, "r") as f:
        dato = json.load(f)
        for key in dato.keys():
            synonyms.append(dato[key])
    
    # Load gold labels from test.gold.txt.
    file_path = os.path.join(folder_path, 'test.gold.txt')
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        nums = [int(line) for line in lines]
    
    # Load test data from test.data.txt.
    file_path = os.path.join(folder_path, 'test.data.txt')
    with open(file_path, 'r', encoding="utf8") as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
        data = [{'number': int(line[0]), 'text': line[1]} for line in lines]
    
    # For each test instance, accumulate sentences:
    #   - The source sentence.
    #   - One candidate sentence per synonym.
    for item in data:
        tokens = item['text'].split()
        nth_token = tokens[item['number']]
        accumulate(synonyms, nth_token, item['text'])
    
    # Each instance contributes (len(synonyms) + 1) sentences.
    block_size = len(synonyms) + 1
    # Use 20% of the folder's test instances (ensuring at least 1 instance)
    num_instances = max(1, int(0.04 * len(data)))
    total_sentences_needed = block_size * num_instances
    sentences = sentences[:total_sentences_needed]
    nums = nums[:num_instances]  # One gold label per instance.
    
    # Get sentence-level embeddings using TF Hub ELMo.
    sentence_embeddings = get_sentence_embeddings(sentences)
    
    # For each block (one source + candidates), pick the candidate with smallest cosine distance.
    for i in range(0, len(sentence_embeddings), block_size):
        block = sentence_embeddings[i:i+block_size]
        if block.shape[0] < block_size:
            break  # Skip incomplete block.
        source_emb = block[0].flatten()
        candidate_embs = block[1:]
        best_distance = float('inf')
        best_candidate_idx = None
        for idx, cand_emb in enumerate(candidate_embs):
            d = distance.cosine(source_emb, cand_emb)
            if d < best_distance:
                best_distance = d
                best_candidate_idx = idx  # This index corresponds to synonyms index.
        results.append(best_candidate_idx)
    
    overall_gold.extend(nums)
    overall_pred.extend(results)
    
    # Clear sentences for the next folder.
    sentences = []

# Compute overall (macro-averaged) precision, recall, and F1-score.
unique_labels = set(overall_gold) | set(overall_pred)
precision_sum = 0.0
recall_sum = 0.0
f1_sum = 0.0

for label in unique_labels:
    TP = sum(1 for gold, pred in zip(overall_gold, overall_pred) if gold == label and pred == label)
    FP = sum(1 for gold, pred in zip(overall_gold, overall_pred) if gold != label and pred == label)
    FN = sum(1 for gold, pred in zip(overall_gold, overall_pred) if gold == label and pred != label)
    p = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    r = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    precision_sum += p
    recall_sum += r
    f1_sum += f

macro_precision = precision_sum / len(unique_labels)
macro_recall = recall_sum / len(unique_labels)
macro_f1 = f1_sum / len(unique_labels)

print('Overall metrics:')
print('  Precision: {:.4f}'.format(macro_precision))
print('  Recall:    {:.4f}'.format(macro_recall))
print('  F1-score:  {:.4f}'.format(macro_f1))

# Compute hits (number of correct predictions) and accuracy.
hits = sum(1 for gold, pred in zip(overall_gold, overall_pred) if gold == pred)
accuracy = hits / len(overall_gold) if overall_gold else 0.0

print('Hits: {}'.format(hits))
print('Accuracy: {:.4f}'.format(accuracy))