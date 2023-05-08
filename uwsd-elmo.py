# -*- coding: utf-8 -*-
"""
[Martinez-Gil2023b] Context-Aware Semantic Similarity Measurement for Unsupervised Word Sense Disambiguation, arXiv preprint arXiv:2305.03520, 2023

@author: Jorge Martinez-Gil
"""

#ELMo
import logging
import os
import json
import numpy as np
from scipy.spatial import distance
from simple_elmo import ElmoModel
  
# Define logging level 
logging.getLogger('simple_elmo').setLevel(logging.ERROR)

# Define the path to the parent directory containing the folders
parent_dir = os.getcwd() + "\\CoarseWSD-20"

# We load the model
model = ElmoModel()

# 144.zip, 193.zip, 225.zip
model.load('225.zip')

# Corpus to work with
sentences = []

def find_token_position(token, lst):
    for i, s in enumerate(lst):
        if token in s:
            return i
    return 0  # Token not found in any string
    
    
def accumulate (synonyms, word, context):

    for i in range(len(synonyms)):
        cons = synonyms[int(i)]
        cons = cons.replace('_', ' ')
        cons = cons.replace('(', '')
        cons = cons.replace(')', '')
        tokens_to_check = cons.split ()
        for token in tokens_to_check:
            if token.lower () == word.lower():
                cons = cons.replace(token, "")
                cons = cons.replace(' ', '')
        
        if word.lower() != cons.lower():
            target = context.replace(word, cons)
            source = context
            if source not in sentences:
                sentences.append (source)
            sentences.append (target)

    return 1


overall_nums = 0
overall_res = 0
overall_baseline = 0
for folder in os.listdir(parent_dir):

    synonyms = []
    nums = []
    data = []
    results = []
    
    # Define the path to the folder
    folder_path = os.path.join(parent_dir, folder)
    
    # Define the path to the file
    file_path = os.path.join(folder_path, 'classes_map.txt')
    with open(file_path, "r") as f:
    
        # Load the data from the file using the json module
        dato = json.load(f)
    
        # Access the values in the data dictionary using their keys
        for key in dato.keys():
            synonyms.append (dato[key])

    file_path = os.path.join(folder_path, 'test.gold.txt')
    with open(file_path, 'r') as f:

        # Read the lines and remove any whitespace characters
        lines = [line.strip() for line in f.readlines()]

        # Convert the lines to integers and store them in a list
        nums = [int(line) for line in lines]


    file_path = os.path.join(folder_path, 'test.data.txt')
    with open(file_path, 'r', encoding="utf8") as f:

        # Read the lines and split them into the number and text sections
        lines = [line.strip().split('\t') for line in f.readlines()]

        # Create a list of dictionaries with keys 'number' and 'text'
        data = [{'number': int(line[0]), 'text': line[1]} for line in lines]


    for item in data:
        r = 0
        tokens = item['text'].split()
        nth_token = tokens[item['number']]
        accumulate (synonyms, nth_token, item['text'])
    
    number = (len(synonyms) + 1)*10 # Corpus is too large, so we take just a fraction
    sentences = sentences[0:number]
    nums = nums[0:number]
    
    m = model.get_elmo_vectors(sentences, layers="average")
            
    fw = 'null'
    maximum = 9999    
    for i in range(len(nums)): 
        list1 = np.array(m[0]).flatten()
        for j in range(len(synonyms)):
            list2 = np.array(m[1]).flatten()
            m = np.delete(m, 1)
            result = distance.cosine(list1, list2)
            if result < maximum:
                fw = synonyms[j]
                maximum = result
        
        m = np.delete(m, 0)
        r = find_token_position(str(fw), list(synonyms)) 
        results.append(r)

    res = sum(x == y for x, y in zip(nums, results))
    
    f = folder_path.split ()
    print (str(f[-1]) + ' result : ' + str(res) + ' in percentage: ' + str(res/len(nums)))
    
    overall_nums = overall_nums + len(nums)
    overall_res = overall_res + res

    count_dict = {}
    for item in nums:
        if item in count_dict:
            count_dict[item] += 1
        else:
            count_dict[item] = 1
    
    sentences = []
    print(count_dict)
    max_value = max(count_dict.values())     
    overall_baseline = overall_baseline + max_value
            
print ('Final result : ' + str(overall_res) + ' in percentage: ' + str(overall_res/overall_nums))
print ('Baseline result : ' + str(overall_baseline) + ' in percentage: ' + str(overall_baseline/overall_nums))