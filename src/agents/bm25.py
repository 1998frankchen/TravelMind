import json
import math
import os
import pickle
import sys
from typing import Dict, List


class BM25:
    EPSILON = 0.25
    PARAM_K1 = 1.5  # BM25 algorithm hyperparameter  
    PARAM_B = 0.6  # BM25 algorithm hyperparameter  
    
    
    def __init__(self, corpus:Dict):
        self.corpus_size = 0  # Number of documents  
        self.wordNumsOfAllDoc = 0  # Used to calculate average word count per document -> wordNumsOfAllDoc / corpus_size  
        self.doc_freqs = {}  # Record word frequency of query terms in each document  
        self.idf = {}  # Record IDF values for query terms  
        self.doc_len = {}  # Record word count for each document  
        
        self.docContainedWord = {}  # Record which documents contain each word  
        
        self._initialize(corpus)
        
        
    def _initialize(self, corpus:Dict):
        """
            Build inverted index based on corpus
        """
        # nd = {} # word -> number of documents containing the word
        for index, document in corpus.items():
            self.corpus_size+=1
            self.doc_len[index] = len(document) # Word count of document  
            self.wordNumsOfAllDoc += len(document) # Total word count of all documents  

            frequencies = {} # Word frequency in one document  
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs[index] = frequencies # Word frequency in one document  
            
            
            # Build inverted index from words to documents, mapping each word to a document set  
            for word in frequencies.keys():
                if word not in self.docContainedWord:
                    self.docContainedWord[word]  = set()
                self.docContainedWord[word].add(index)
                    
                
        # Calculate IDF  
        idf_sum = 0  # Used to calculate IDF denominator  
        negative_idfs = []
        # Calculate original IDF
        # idf = log((total_docs - docs_containing_word + 0.5) / (docs_containing_word + 0.5))

        # # Negative value correction strategy
        # average_idf = total_idf / word_count
        # eps = EPSILON * average_idf
        # self.idf[negative_idf_word] = eps  # Replace negative values with 25% of average IDF    
        
        for word in self.docContainedWord.keys():
            doc_nums_contained_word = len(self.docContainedWord[word])
            idf = math.log(self.corpus_size-doc_nums_contained_word + 0.5) - \
                math.log(doc_nums_contained_word + 0.5)
            
            self.idf[word] = idf        
            idf_sum+=idf
            
            if idf < 0:
                negative_idfs.append(word)
        
        
        average_idf = idf_sum / len(self.idf)
        eps = BM25.EPSILON * average_idf
        
        for word in negative_idfs:
            self.idf[word] = eps
            
        
        print("==============================")
        print("self.doc_freqs",[f"doc_id:{k}, " for k, v in self.doc_freqs.items()])
        print("self.idf", self.idf)
        print("self.doc_len", self.doc_len)
        print("self.docContainedWord", self.docContainedWord)
        print("self.corpus_size", self.corpus_size)
        print("========================================")

    
    @property
    def avgdl(self):
        return self.wordNumsOfAllDoc / self.corpus_size
    
    
    
    
    def get_score(self, query:List, doc_index):
        """
        Calculate relevance score between query q and document d
        :param query: List of query terms
        :param doc_index: Index of a document in the corpus

        score(D, Q) = Σ IDF(q_i) * [ (f(q_i,D) * (k1 + 1)) / (f(q_i,D) + k1*(1 - b + b*|D|/avgdl)) ]
        """
        score = 0
        b = BM25.PARAM_B
        k1 = BM25.PARAM_K1
        avgdl = self.avgdl
        
        doc_freqs = self.doc_freqs[doc_index]
        
        for word in query:
            if word not in doc_freqs:
                continue
            score += self.idf[word] * (doc_freqs[word] * (k1+1) / (doc_freqs[word] + k1*(1-b+b*self.doc_len[doc_index]/avgdl)))
        
        return [doc_index, score]
        
    def get_scores(self, query):
        scores = [self.get_score(query, index) for index in self.doc_len.keys()]
        return scores
