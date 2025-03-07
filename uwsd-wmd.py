# -*- coding: utf-8 -*-
"""
[Martinez-Gil2023b]  Context-Aware Semantic Similarity Measurement for Unsupervised Word Sense Disambiguation, arXiv preprint arXiv:2305.03520, 2023

@author: Jorge Martinez-Gil
"""

import os
import json
import random
import torch
import gensim.downloader as api
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

# Check for CUDA (note: gensim models do not use CUDA for WMD)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------
# Helper Functions
# -----------------------

def find_token_position(token, lst):
    """
    Return the index of the first string in lst that contains token.
    """
    for i, s in enumerate(lst):
        if token in s:
            return i
    return 0  # if token not found

def calculate_candidate_wmd(model, synonyms, word, context):
    """
    For each candidate synonym, clean it and replace the target word in the context.
    Compute the Word Mover's Distance (WMD) between the candidate sentence and the original context.
    Return the candidate with the minimum distance.
    """
    best_candidate = 'null'
    min_distance = float('inf')
    
    for cons in synonyms:
        # Clean the candidate synonym
        candidate = cons.replace('_', ' ').replace('(', '').replace(')', '')
        tokens_to_check = candidate.split()
        for token in tokens_to_check:
            if token.lower() == word.lower():
                candidate = candidate.replace(token, "")
        # Skip if candidate becomes empty or equals the original word
        if candidate.strip() == "" or candidate.lower() == word.lower():
            continue
        
        # Create a candidate sentence by replacing the target word with the candidate
        source = context.replace(word, candidate)
        target = context
        try:
            distance = model.wmdistance(source, target)
        except Exception as e:
            # In case of any error during distance computation, skip this candidate.
            continue
        if distance < min_distance:
            best_candidate = candidate
            min_distance = distance

    return best_candidate

def evaluate_method(method_name, gensim_model_name, parent_dir):
    """
    Evaluate one method over all folders.
    
    For model-based methods, load the gensim model (using gensim.downloader).
    For baselines (Most Frequent Sense and Random Choice), gensim_model_name should be None.
    
    Returns a dictionary with overall hits, accuracy, precision, recall, and F1.
    """
    all_gold = []
    all_preds = []
    total_hits = 0
    total_samples = 0

    # Load gensim model if needed
    model = None
    if gensim_model_name is not None:
        print(f"Loading model: {gensim_model_name}")
        model = api.load(gensim_model_name)
        # For older versions: if available, initialize sims to speed up similarity computations.
        if hasattr(model, "init_sims"):
            model.init_sims(replace=True)
    
    # Loop over each folder in the dataset directory
    folder_list = sorted(os.listdir(parent_dir))
    for folder in folder_list:
        folder_path = os.path.join(parent_dir, folder)
        
        # Load synonyms from classes_map.txt
        synonyms_path = os.path.join(folder_path, 'classes_map.txt')
        with open(synonyms_path, "r", encoding="utf8") as f:
            data_syn = json.load(f)
            # Assume the order of synonyms follows the order of the keys.
            synonyms = [data_syn[key] for key in data_syn.keys()]
        
        # Load gold labels from test.gold.txt
        gold_path = os.path.join(folder_path, 'test.gold.txt')
        with open(gold_path, 'r') as f:
            gold_labels = [int(line.strip()) for line in f.readlines()]
        
        # Load test data from test.data.txt
        test_path = os.path.join(folder_path, 'test.data.txt')
        with open(test_path, 'r', encoding="utf8") as f:
            lines = [line.strip().split('\t') for line in f.readlines()]
            test_data = [{'number': int(line[0]), 'text': line[1]} for line in lines]
        
        # For Most Frequent Sense baseline: determine the majority gold label for the folder
        if method_name == "Most Frequent Sense":
            majority_label = max(set(gold_labels), key=gold_labels.count)
        
        # Process each test sample
        for item in test_data:
            text = item['text']
            tokens = text.split()
            target_word = tokens[item['number']]
            
            if method_name == "Most Frequent Sense":
                pred_label = majority_label
            elif method_name == "Random Choice":
                pred_label = random.choice(range(len(synonyms)))
            else:
                # Model-based method using WMD
                candidate = calculate_candidate_wmd(model, synonyms, target_word, text)
                # Remove spaces to match the format in synonyms
                candidate_clean = candidate.replace(" ", "")
                pred_label = find_token_position(candidate_clean, synonyms)
            
            # For evaluation, assume the gold labels follow the order in gold_labels.
            # Pop the first gold label (each folder's labels are in order).
            gold = gold_labels.pop(0)
            all_gold.append(gold)
            all_preds.append(pred_label)
            total_samples += 1
            if pred_label == gold:
                total_hits += 1

    # Compute overall performance metrics
    accuracy = accuracy_score(all_gold, all_preds)
    precision = precision_score(all_gold, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_gold, all_preds, average='weighted', zero_division=0)
    # Manually calculate F1 measure from precision and recall
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

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

# Set the parent directory containing the dataset folders.
# Adjust the path as needed.
parent_dir = os.path.join(os.getcwd(), "CoarseWSD-20")

# Define the list of methods and corresponding gensim model names.
# For baselines, the gensim model name is None.
methods = [
    ("Most Frequent Sense", None),
    ("glove-twitter-200", "glove-twitter-200"),
    ("word2vec-google-new-300", "word2vec-google-news-300"),
    ("glove-wiki-gigaword-300", "glove-wiki-gigaword-300"),
    ("fasttext-wiki-news-subwords-300", "fasttext-wiki-news-subwords-300"),
    ("Random Choice", None)
]

results_list = []
for method_name, gensim_model_name in methods:
    print(f"\nEvaluating method: {method_name}")
    res = evaluate_method(method_name, gensim_model_name, parent_dir)
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
latex_table += r"\caption{Evaluation Results for WMD-based Methods in Unsupervised Word Sense Disambiguation}" + "\n"
latex_table += r"\label{tab:uwsd_results}" + "\n"
latex_table += r"\end{table}"

print("\nLaTeX Table:")
print(latex_table)
