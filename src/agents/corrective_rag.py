from typing import List, Tuple, Dict  
from langchain_core.documents import Document  
from langchain_community.retrievers.bm25 import BM25Retriever  
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from zhipuai import ZhipuAI  
import os  
import requests  


'''
CorrectiveRAG is not yet complete and cannot run.
'''

class CorrectiveRAG:  
    def __init__(self,   
                 corpus: List[str] = None,
                 corpus_folder_path = None,
                 model_type: str = "api",  
                 correct_threshold: float = 0.59,  
                 incorrect_threshold: float = -0.99):  
        
        assert corpus is not None or corpus_folder_path is not None, "Either corpus or corpus_folder_path must be provided."

        
        if corpus==None:
            self.corpus = self.load_corpus(corpus_folder_path)
        else:
            self.corpus = corpus
        
        self.client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))  
        self.retriever = BM25Retriever.from_texts(self.corpus)  
        
        # Recursive character text splitter used to split long documents into smaller chunks.
        # In the knowledge_refinement method, it splits retrieved documents into smaller segments for more detailed relevance evaluation.  
        self.text_splitter = RecursiveCharacterTextSplitter(  
            chunk_size=200,   # Maximum length of each text chunk is 200 characters
            chunk_overlap=50   # 50 characters overlap between adjacent text chunks  
        )  
        self.model_type = model_type  
        self.thresholds = {  
            'correct': correct_threshold,  
            'incorrect': incorrect_threshold  
        }  
        
    
    
    def load_corpus(self, corpus_folder_path):
        pass
    
    
    

    def evaluate_relevance(self, query: str, document: str) -> float:
        """Use Zhipu API to evaluate relevance between query and document"""    
        response = self.client.chat.completions.create(  
            model="glm-4",  
            messages=[{  
                "role": "user",  
                "content": f"Please evaluate the relevance between the following query and document, return a score between 0-1:\nQuery: {query}\nDocument: {document}"
            }]  
        )  
        try:  
            return float(response.choices[0].message.content.strip())  
        except:  
            print("\nRelevance score is 0, please check the following content:")
            print(response.choices[0].message.content)
            print()
            return 0.0  

    def retrieve_documents(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents"""    
        return self.retriever.invoke(query)[:k]  

    def knowledge_refinement(self, query: str, documents: List[Document]) -> str:  
        """Knowledge精炼Process"""    
        refined = []  
        for doc in documents:  
            chunks = self.text_splitter.split_text(doc.page_content)  
            for chunk in chunks:  
                score = self.evaluate_relevance(query, chunk)  
                if score > 0.5:  # Filter阈Value    
                    refined.append(chunk)  
        return "\n\n".join(refined[:3])  # 保留前3个相关片段    

    def web_search(self, query: str) -> str:  
        """NetworkSearchEnhancement"""  
        # Query改写    
        rewrite_prompt = f"将以Down/BelowQuestion/ProblemTransform为Search关Key词（3-5个关Key词，用分号分隔）：{query}"    
        keywords = self.client.chat.completions.create(  
            model="glm-4",  
            messages=[{"role": "user", "content": rewrite_prompt}]  
        ).choices[0].message.content.split(";")  
        
        # ExecuteSearch（ExampleUsageSerper API）  
        headers = {  
            'X-API-KEY': os.getenv("SERPER_API_KEY"),  
            'Content-Type': 'application/json'  
        }  
        response = requests.post(  
            'https://google.serper.dev/search',  
            headers=headers,  
            json={'q': " ".join(keywords)}  
        )  
        return "\n".join([result.get("snippet", "") for result in response.json().get("organic", [])[:3]])  

    def determine_action(self, scores: List[float]) -> str:  
        """确定ExecuteAction"""    
        max_score = max(scores)  
        min_score = min(scores)  
        
        if max_score > self.thresholds['correct']:  
            return "correct"  
        elif min_score < self.thresholds['incorrect']:  
            return "incorrect"  
        else:  
            return "ambiguous"  

    def generate_response(self, query: str, context: str) -> str:  
        """Generate最终Response"""    
        response = self.client.chat.completions.create(  
            model="glm-4",  
            messages=[{  
                "role": "user",  
                "content": f"基于以Down/BelowContext回答Question/Problem：\n{context}\n\nQuestion/Problem：{query}"    
            }]  
        )  
        return response.choices[0].message.content  

    def run(self, query: str) -> str:  
        # 检索Stage    
        documents = self.retrieve_documents(query)  
        
        # Evaluate/EvaluationStage  
        scores = [self.evaluate_relevance(query, doc.page_content) for doc in documents]  
        action = self.determine_action(scores)  
        
        # KnowledgeProcess  
        if action == "correct":  
            knowledge = self.knowledge_refinement(query, documents)  
        elif action == "incorrect":  
            knowledge = self.web_search(query)  
        else:  
            refined = self.knowledge_refinement(query, documents)  
            web_result = self.web_search(query)  
            knowledge = f"{refined}\n\n{web_result}"  
        
        # GenerateStage  
        return self.generate_response(query, knowledge)  

 
if __name__ == "__main__":  
    corpus = [  
        # Add更多Documentation...    
    ]  
    
    
    document_path = "src\\agents\\travel_knowledge\\tour_pages"
    crag = CorrectiveRAG(corpus_folder_path=document_path)  
    
    query = "请帮我规划一个杭州一Day游Route"    
    response = crag.run(query)  
    print(f"Question/Problem：{query}\n回答：{response}")    