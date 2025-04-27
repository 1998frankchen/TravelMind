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
    
    
    





BM25_ALGORITHM_DESC = """

### BM25算法原理详解  

BM25（Best Matching 25）Yes/IsInformation检索Domain广泛Usage的相关性Score/Rating算法，Core思想Yes/IsPass概率ModelEvaluate/EvaluationQuery与Documentation的相关性。其Core公式为：  

```
score(D, Q) = Σ IDF(q_i) * [ (f(q_i,D) * (k1 + 1)) / (f(q_i,D) + k1*(1 - b + b*|D|/avgdl)) ]
```

**Core要素**：  
1. **IDF（逆Documentation频率）**：衡量词语区分能力  
   - 公式：`IDF(q_i) = log[(N - n(q_i) + 0.5)/(n(q_i) + 0.5) + 1]`  
   - 其MediumNYes/Is总Documentation数，n(q_i)Yes/IsPackage含q_i的Documentation数  

2. **TF（词频）调整**：PassParameterk1Control词频饱和度  
   - 当k1=0时完全Ignore词频  
   - 典型ValueScope：1.2~2.0  

3. **Length归一化**：PassParameterb平衡DocumentationLength影响  
   - b=1时完全归一化，b=0时IgnoreLength  
   - Usage`|D|/avgdl`计算相对Length  
   
   
   BM25的本质：  
        它就Yes/IsqueryMedium每个词的idf分数的累加  
   

### 代码ClassParse  

```python
class BM25:
    # ClassConstantDefinition
    EPSILON = 0.25  # 负IDFAmend系数  
    PARAM_K1 = 1.5  # 词频饱和度Parameter  
    PARAM_B = 0.6   # Length归一化Parameter  

    def __init__(self, corpus: Dict):
        # InitializeData结构  
        self.corpus_size = 0          # Documentation总数  
        self.wordNumsOfAllDoc = 0     # 语料Library总词数  
        self.doc_freqs = {}           # {DocumentationID: {词: 词频}}  
        self.idf = {}                 # {词: IDFValue}  
        self.doc_len = {}             # {DocumentationID: 词数}  
        self.docContainedWord = {}    # {词: Package含该词的DocumentationSet}  
```

#### 关KeyMethodParse  

1. **倒排IndexBuild** (`_initialize`)  
```python
for index, document in corpus.items():
    # StatisticsDocumentationLength
    self.doc_len[index] = len(document)
    
    # Build词频Dictionary/Dict  
    frequencies = {}
    for word in document:
        frequencies[word] = frequencies.get(word, 0) + 1
    
    # Build倒排Index  
    for word in frequencies:
        self.docContainedWord.setdefault(word, set()).add(index)
```

2. **IDF计算Optimize**  
```python
# 计算原始IDF  
idf = log((总Documentation数 - Package含词数 + 0.5) / (Package含词数 + 0.5))  

# 负ValueAmendStrategy  
average_idf = 总IDF / 词数  
eps = EPSILON * average_idf
self.idf[负IDF词] = eps  # 用平均IDF的25%替代负Value  
```

3. **相关性计算** (`get_score`)  
```python
for word in query:
    if word in Documentation词频:  
        # 计算TF分量  
        tf = doc_freqs[word]
        # 计算Length归一化因子  
        norm_factor = 1 - b + b*(DocumentationLength/平均Length)  
        # 累加词项得分  
        score += idf * (tf*(k1+1)) / (tf + k1*norm_factor)
```

### Parameter作用Instruction/Description  

| Parameter   | 典型ValueScope | Function/FeatureInstruction/Description                                                                 |  
|--------|------------|--------------------------------------------------------------------------|
| k1     | 1.2-2.0    | Control词频饱和度：Value越大，High频词影响越大                                   |  
| b      | 0.5-0.8    | Length归一化强度：1表示完全归一化，0DisableLength调整                           |  
| EPSILON| 0.2-0.3    | 负IDFAmend系数：防止罕见词产生负Weight，Keep/Maintain数ValueStable性                      |  

该ImplementationComplete/IntactPackage含了BM25的Core要素，Pass预计算倒排Index和IDFValueImplementationHigh效检索，适合Medium小规模语料Library的Real-timeSearch场景。  







"""