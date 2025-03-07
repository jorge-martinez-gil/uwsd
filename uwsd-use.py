# -*- coding: utf-8 -*-
"""
[Martinez-Gil2023b]  Context-Aware Semantic Similarity Measurement for Unsupervised Word Sense Disambiguation, arXiv preprint arXiv:2305.03520, 2023

@author: Jorge Martinez-Gil
"""

import os
import json
import random
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tensorflow_hub as hub
from tqdm import tqdm

# Set the parent directory containing the dataset folders.
parent_dir = os.path.join(os.getcwd(), "CoarseWSD-20")

# Load the Universal Sentence Encoder models.
print("Loading USE Classic...")
embed_classic = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print("Loading USE Large...")
embed_large = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

def find_token_position(token, lst):
    """
    Return the index of the first string in lst that contains token.
    If not found, return 0.
    """
    token_clean = token.lower().replace(" ", "")
    for i, s in enumerate(lst):
        if token_clean in s.lower().replace(" ", ""):
            return i
    return 0

def calculate_candidate_use(synonyms, word, context, embed):
    """
    For each candidate synonym in 'synonyms', clean the candidate and replace the target word
    in the context. Then compute the cosine distance between the candidate sentence and the
    original sentence using the provided USE embedder.
    Return the candidate with the smallest distance.
    """
    best_candidate = 'null'
    min_distance = float('inf')
    
    for cons in synonyms:
        # Clean candidate: remove underscores and parentheses.
        candidate = cons.replace('_', ' ').replace('(', '').replace(')', '')
        # Remove tokens matching the target word.
        candidate_tokens = candidate.split()
        candidate_clean = " ".join([tok for tok in candidate_tokens if tok.lower() != word.lower()])
        
        # Skip if candidate becomes empty or equals the target word.
        if not candidate_clean.strip() or candidate_clean.lower() == word.lower():
            continue
        
        # Create candidate sentence by replacing the target word.
        candidate_sentence = context.replace(word, candidate_clean)
        # Compute USE embeddings for candidate sentence and original context.
        embeddings = embed([candidate_sentence, context])
        cos_dist = distance.cosine(embeddings[0], embeddings[1])
        
        if cos_dist < min_distance:
            best_candidate = candidate_clean
            min_distance = cos_dist

    return best_candidate

def evaluate_method(method_name, use_embed, parent_dir):
    """
    Evaluate one method over all folders in CoarseWSD-20.
    
    For USE methods, use_embed is provided (embed_classic or embed_large).
    For baselines, use_embed is None.
    
    Returns a dictionary with overall hits, accuracy, precision, recall, and F1.
    """
    all_gold = []
    all_preds = []
    total_hits = 0
    total_samples = 0

    # Loop over each folder in the dataset.
    folder_list = sorted(os.listdir(parent_dir))
    for folder in tqdm(folder_list, desc=f"Evaluating {method_name}"):
        folder_path = os.path.join(parent_dir, folder)
        
        # Load synonyms from classes_map.txt.
        syn_path = os.path.join(folder_path, 'classes_map.txt')
        with open(syn_path, "r", encoding="utf8") as f:
            data_syn = json.load(f)
            # Assume order follows keys order.
            synonyms = [data_syn[key] for key in data_syn.keys()]
        
        # Load gold labels from test.gold.txt.
        gold_path = os.path.join(folder_path, 'test.gold.txt')
        with open(gold_path, 'r') as f:
            gold_labels = [int(line.strip()) for line in f.readlines()]
        
        # Load test data from test.data.txt.
        test_path = os.path.join(folder_path, 'test.data.txt')
        with open(test_path, 'r', encoding="utf8") as f:
            lines = [line.strip().split('\t') for line in f.readlines()]
            test_data = [{'number': int(line[0]), 'text': line[1]} for line in lines]
        
        # For "Most Frequent Sense" baseline: compute the majority label.
        if method_name == "Most Frequent Sense":
            majority_label = max(set(gold_labels), key=gold_labels.count)
        
        # Process each test instance.
        for item in test_data:
            text = item['text']
            tokens = text.split()
            target_word = tokens[item['number']]
            
            if method_name == "Most Frequent Sense":
                pred_label = majority_label
            elif method_name == "Random Choice":
                pred_label = random.choice(range(len(synonyms)))
            else:  # For USE Classic or USE Large.
                candidate = calculate_candidate_use(synonyms, target_word, text, use_embed)
                candidate_clean = candidate.replace(" ", "")
                pred_label = find_token_position(candidate_clean, synonyms)
            
            # Get the next gold label.
            gold = gold_labels.pop(0)
            all_gold.append(gold)
            all_preds.append(pred_label)
            total_samples += 1
            if pred_label == gold:
                total_hits += 1

    accuracy = accuracy_score(all_gold, all_preds)
    precision = precision_score(all_gold, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_gold, all_preds, average='macro', zero_division=0)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "model": method_name,
        "hits": total_hits,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# -----------------------
# Main Evaluation
# -----------------------

# Define the list of methods.
# For USE methods, provide the corresponding embed; for baselines, use None.
methods = [
    ("Most Frequent Sense", None),
    ("USE Classic", embed_classic),
    ("USE Large", embed_large),
    ("Random Choice", None)
]

results_list = []
for method_name, use_embed in methods:
    print(f"\nEvaluating method: {method_name}")
    res = evaluate_method(method_name, use_embed, parent_dir)
    results_list.append(res)
    print(f"Method: {method_name}, Hits: {res['hits']}, Accuracy: {res['accuracy']:.4f}, "
          f"Precision: {res['precision']:.4f}, Recall: {res['recall']:.4f}, F1: {res['f1']:.4f}")

# -----------------------
# Produce LaTeX Table
# -----------------------

latex_table = r"\begin{table}[ht]" + "\n"
latex_table += r"\centering" + "\n"
latex_table += r"\begin{tabular}{lccccc}" + "\n"
latex_table += r"\hline" + "\n"
latex_table += r"Model & Hits & Accuracy & Precision & Recall & F1 \\" + "\n"
latex_table += r"\hline" + "\n"

for res in results_list:
    acc_str = f"{res['accuracy']*100:.2f}\\%"
    prec_str = f"{res['precision']*100:.2f}\\%"
    rec_str = f"{res['recall']*100:.2f}\\%"
    f1_str = f"{res['f1']*100:.2f}\\%"
    line = f"{res['model']} & {res['hits']} & {acc_str} & {prec_str} & {rec_str} & {f1_str} \\\\"
    latex_table += line + "\n"

latex_table += r"\hline" + "\n"
latex_table += r"\end{tabular}" + "\n"
latex_table += r"\caption{Evaluation Results on CoarseWSD-20 using USE Classic and USE Large}" + "\n"
latex_table += r"\label{tab:use_classic_large}" + "\n"
latex_table += r"\end{table}"

print("\nLaTeX Table:")
print(latex_table)
