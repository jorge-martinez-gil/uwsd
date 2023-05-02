# -*- coding: utf-8 -*-
"""
[Martinez-Gil2023b] Context-Aware Semantic Similarity Measurement for Unsupervised Word Sense Disambiguation, arXiv preprint arXiv:2301.00xxx, 2023

@author: Jorge Martinez-Gil
"""

# Modules
import requests
from sentence_transformers import SentenceTransformer, util
    
def calculate(word, context):
    
    fw = 'null'
    model = SentenceTransformer('all-MiniLM-L12-v2')
    
    synonyms = []
    str1 = 'https://tuna.thesaurus.com/pageData/' + str(word)
    req = requests.get(str1)
    try:
        dict_synonyms = req.json()['data']['definitionData']['definitions'][0]['synonyms']
    except TypeError as e:
        print ("Processing...Please wait")
        dict_synonyms = None
                 
    if dict_synonyms is not None:
        synonyms = [r["term"] for r in dict_synonyms]
        
    maximum = 9999
    for i in range(len(synonyms)):
        cons = synonyms[int(i)]
        if word.lower() != cons.lower():
            source = context.replace(word, cons)
            source = source.replace('_', ' ')
            target = context
            source_embedding = model.encode(source)
            target_embedding = model.encode(target)
            result0 = util.cos_sim(source_embedding, target_embedding)
            resulta = [float(t.item()) for t in result0]
            result = 1-resulta[0]
            # print ('Comparing ' + source + ' <-> ' + target + ' ' + str(result))
            if result < maximum:
                fw = cons
                maximum = result
        
    return fw
            
text = 'Linz is a nice city in the heart of Europe'
fr = calculate ('nice', text)
print (fr)
        

