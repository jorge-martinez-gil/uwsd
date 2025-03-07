# -*- coding: utf-8 -*-
"""
[Martinez-Gil2023b]  Context-Aware Semantic Similarity Measurement for Unsupervised Word Sense Disambiguation, arXiv preprint arXiv:2305.03520, 2023

@author: Jorge Martinez-Gil
"""

# Install required packages if needed:
# !pip install --upgrade sentence-transformers huggingface_hub scikit-learn

import os
import json
import torch
import random
from sentence_transformers import SentenceTransformer, util
from tqdm.notebook import tqdm
from sklearn.metrics import precision_score, recall_score
import numpy as np

# Check GPU availability and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ----- Helper Functions -----
def calculate_candidate(model, synonyms, word, context):
    """
    For each candidate synonym, replace the target word in the context,
    compute the cosine similarity in a batch, and return the candidate 
    (as a string) with the highest similarity.
    """
    processed_candidates = []
    candidate_sentences = []
    
    for syn in synonyms:
        # Clean candidate: remove underscores and parentheses.
        candidate = syn.replace('_', ' ').replace('(', '').replace(')', '')
        # Remove any token that matches the original word.
        candidate_tokens = candidate.split()
        candidate_clean = " ".join([tok for tok in candidate_tokens if tok.lower() != word.lower()])
        
        # Skip if candidate becomes empty or equals the original word.
        if not candidate_clean.strip() or candidate_clean.lower() == word.lower():
            continue
        
        # Replace the target word in the context with the candidate.
        candidate_sentence = context.replace(word, candidate_clean)
        processed_candidates.append(candidate_clean)
        candidate_sentences.append(candidate_sentence)
    
    if not candidate_sentences:
        return 'null'
    
    # Precompute the target context embedding.
    target_emb = model.encode(context, convert_to_tensor=True)
    # Batch encode candidate sentences.
    candidate_embs = model.encode(candidate_sentences, convert_to_tensor=True)
    # Compute cosine similarities.
    scores = util.cos_sim(candidate_embs, target_emb).squeeze()  # shape: (num_candidates,)
    best_idx = torch.argmax(scores).item()
    best_candidate = processed_candidates[best_idx]
    return best_candidate

def find_token_position(token, lst):
    """
    Given a token and a list of synonyms, return the index of the first synonym
    that contains the token (ignoring spaces and case). If not found, return 0.
    """
    token_clean = token.lower().replace(" ", "")
    for i, s in enumerate(lst):
        if token_clean in s.lower().replace(" ", ""):
            return i
    return 0

# ----- Evaluation Function -----
def evaluate_method(method_name, model_name, parent_dir):
    """
    Evaluate one method over all folders.
    For model-based methods (when model_name is not None), load the corresponding model.
    For baseline methods ("Most frequent sense" and "random choice"), no model is loaded.
    Returns a dict with overall hits, accuracy, precision, recall, and F1.
    """
    all_gold = []
    all_preds = []
    total_hits = 0
    total_samples = 0
    
    # Load the SentenceTransformer model if needed.
    if model_name is not None:
        current_model = SentenceTransformer(model_name, device=device)
    else:
        current_model = None

    # Loop over each folder in the dataset directory.
    for folder in tqdm(os.listdir(parent_dir), desc=f"Evaluating {method_name}"):
        folder_path = os.path.join(parent_dir, folder)
        
        # Load synonyms from classes_map.txt.
        with open(os.path.join(folder_path, 'classes_map.txt'), "r") as f:
            dato = json.load(f)
            # The synonyms list order is based on the keys order.
            synonyms = [dato[key] for key in dato.keys()]
        
        # Load gold labels.
        with open(os.path.join(folder_path, 'test.gold.txt'), 'r') as f:
            gold_labels = [int(line.strip()) for line in f.readlines()]
        
        # Load test samples.
        with open(os.path.join(folder_path, 'test.data.txt'), 'r', encoding="utf8") as f:
            lines = [line.strip().split('\t') for line in f.readlines()]
            data = [{'number': int(line[0]), 'text': line[1]} for line in lines]
        
        # For "Most frequent sense", compute the majority (mode) gold label for this folder.
        if method_name == "Most frequent sense":
            majority_label = max(set(gold_labels), key=gold_labels.count)
        
        # Process each test sample.
        for item in data:
            text = item['text']
            tokens = text.split()
            # The target word is at the given index.
            target_word = tokens[item['number']]
            
            # Choose prediction based on the method.
            if method_name == "Most frequent sense":
                pred_label = majority_label
            elif method_name == "Random choice":
                pred_label = random.choice(range(len(synonyms)))
            else:
                # For model-based methods, use the candidate selection function.
                candidate = calculate_candidate(current_model, synonyms, target_word, text)
                # Remove spaces to match the format in synonyms.
                candidate_clean = candidate.replace(" ", "")
                pred_label = find_token_position(candidate_clean, synonyms)
            
            all_gold.append(gold_labels.pop(0))  # Gold labels are in order.
            all_preds.append(pred_label)
            total_samples += 1
            if all_preds[-1] == all_gold[-1]:
                total_hits += 1

    # Compute macro precision, recall, and F1.
    precision = precision_score(all_gold, all_preds, average='macro')
    recall = recall_score(all_gold, all_preds, average='macro')
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = total_hits / total_samples if total_samples > 0 else 0

    return {
        "model": method_name,
        "hits": total_hits,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ----- Main Evaluation Over All Methods -----
# Define the dataset directory path (adjust if needed)
parent_dir = "/content/drive/MyDrive/datasets/CoarseWSD20"

# Define the list of methods along with their model names.
# For baseline methods, model_name is None.
methods = [
    ("all-mpnet-base-v2", "all-mpnet-base-v2"),
    ("all-MiniLM-L12-v2", "all-MiniLM-L12-v2"),
    ("all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
    ("Most frequent sense", None),
    ("paraphrase-albert-small-v2", "paraphrase-albert-small-v2"),
    ("paraphrase-MiniLM-L3-v2", "paraphrase-MiniLM-L3-v2"),
    ("all-distilroberta-v1", "all-distilroberta-v1"),
    ("Random choice", None)
]

results_list = []
for method_name, model_name in methods:
    res = evaluate_method(method_name, model_name, parent_dir)
    results_list.append(res)

# ----- Produce LaTeX Table -----
latex_table = r"\begin{table}[ht]" + "\n"
latex_table += r"\centering" + "\n"
latex_table += r"\begin{tabular}{lccccc}" + "\n"
latex_table += r"\hline" + "\n"
latex_table += r"Model & Hits & Accuracy & Precision & Recall & F1 \\" + "\n"
latex_table += r"\hline" + "\n"

for res in results_list:
    # Format percentages with two decimals.
    acc = f"{res['accuracy']*100:.2f}\\%"
    prec = f"{res['precision']*100:.2f}\\%"
    rec = f"{res['recall']*100:.2f}\\%"
    f1_val = f"{res['f1']*100:.2f}\\%"
    line = f"{res['model']} & {res['hits']} & {acc} & {prec} & {rec} & {f1_val} \\\\"
    latex_table += line + "\n"

latex_table += r"\hline" + "\n"
latex_table += r"\end{tabular}" + "\n"
latex_table += r"\caption{Evaluation Results for Different Models and Baselines}" + "\n"
latex_table += r"\label{tab:evaluation_results}" + "\n"
latex_table += r"\end{table}"

print(latex_table)