# -*- coding: utf-8 -*-
"""
[Martinez-Gil2023b] Context-Aware Semantic Similarity Measurement for Unsupervised Word Sense Disambiguation, arXiv preprint arXiv:2301.00xxx, 2023

@author: Jorge Martinez-Gil
"""

# Modules
import gensim.downloader
from sentence_transformers import SentenceTransformer, util
google_news_vectors = gensim.downloader.load('word2vec-google-news-300')
      
def calculate(word, context):
    
    fw = 'null'
    model = SentenceTransformer('all-MiniLM-L12-v2')
    synonyms = google_news_vectors.most_similar(word, topn=8)
    
    maximum = 9999
    for i in range(len(synonyms)):
        cons = synonyms[int(i)]
        cons = str(cons[0])
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
        

