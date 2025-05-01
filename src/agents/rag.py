


try:

    from src.agents.prompt_template import MyPromptTemplate
    from src.agents.tools import ToolDispatcher
    from typing import Dict, List, Optional, Tuple
    from src.models.model import TravelMind
    from src.data.data_processor import CrossWOZProcessor

except Exception as e:
    print("Import error occurred:", str(e))
    print("================================")
    
    

try:

    from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain  
    from langchain.memory.buffer import ConversationBufferMemory  
    from langchain_community.vectorstores.chroma import Chroma      # pip install langchain-chroma  pip install langchain_community
    from langchain_core.prompts import ChatPromptTemplate  
    from langchain_core.runnables import RunnablePassthrough  
    from langchain_core.output_parsers import StrOutputParser  
    from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings  
    from langchain_core.prompts import PromptTemplate

    from langchain.graphs import Neo4jGraph  
    from langchain.chains.graph_qa.cypher import GraphCypherQAChain
    
    from datasets import Dataset
    # from langchain_experimental.graph_transformers import ChainMap     # pip install langchain_experimental
except Exception as e:
    print("langchain import error:", str(e))  
    print("=================================")
    
    

'''
Suggest/RecommendCheckPydanticVersionCompatible性，RecommendedUsage：  

pip install pydantic>=2.5.0  

'''

from datasets import load_dataset
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction  
import re
import os
import json
import torch
import numpy as np
import jieba
from zhipuai import ZhipuAI 

from .bm25 import BM25

from src.configs.config import RAG_DATA_PATH, SFT_MODEL_PATH, EMBEDDING_MODEL_PATH, PAGE_FOLDER_PATH


ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")


class LocalEmbeddingFunction(EmbeddingFunction):
    """Local embedding model adapter"""    
    def __init__(self, model_name: str = EMBEDDING_MODEL_PATH):  
        self.embedder = HuggingFaceEmbeddings(  
            model_name=model_name,  
            model_kwargs={'device': 'cpu'},  
            encode_kwargs={'normalize_embeddings': True}  
        )  

    def __call__(self, texts: List[str]) -> List[List[float]]:  
        return self.embedder.embed_documents(texts)  









class RAG():
    def __init__(
        self, 
        agent: TravelMind,
        dataset_name_or_path:str = RAG_DATA_PATH,
        embedding_model_name_or_path:str = EMBEDDING_MODEL_PATH,
        use_langchain = False,
        use_prompt_template = True,
        use_db = True,
        use_api=False
        ):
        self.use_langchain = use_langchain
        self.use_prompt_template = use_prompt_template
        self.use_db = use_db
        self.agent = agent
        self.use_api = use_api
        
        if use_db:
            self.embedding_fn = LocalEmbeddingFunction(model_name=embedding_model_name_or_path)
            self.chroma_client = chromadb.Client()
            print("Chroma dataset construction completed")  
            # self.chroma_client = chromadb.PersistentClient(path = "local dir")
            
            self.collection = self.chroma_client.create_collection(
                name="my_collection",
                embedding_function=self.embedding_fn,
                metadata={
                    "hnsw:space":"cosine",
                    "embedding_model": embedding_model_name_or_path
                })
            print("Chroma data table construction completed")  
            self.dataset =  load_dataset(dataset_name_or_path, split="train").select(range(1000))
            print("Crosswoz dataset loading completed~~~")  
            self.embeddings = LocalEmbeddingFunction(EMBEDDING_MODEL_PATH).embedder
            print("Embedding model loading completed~~~~")  
            # Automatically generate embeddings when loading dataset    
            self._initialize_database() 
            print("Crosswoz dataset successfully converted to embedding vectors.")  
        
        
        if self.use_prompt_template:
            self.prompt_template = MyPromptTemplate()
            self.dispatcher = ToolDispatcher()
    
    
    # def parse_db(self):
    #     assert self.use_db, "The embedding database is not initialized."
    #     result = []
        
    #     for sample in self.dataset:
    #         result.append(sample["history"])
            
    #     return result
    
    def call_api_model(self, prompt):
        client = ZhipuAI(api_key=ZHIPU_API_KEY) 
        response = client.chat.completions.create(
            model="glm-4-flash",  # 填写需要调用的ModelName  
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        response_text = response.choices[0].message.content
        return response_text
    
    
    
    
    def _initialize_database(self, field = "history"):
        """Initialize data library using local embedding model"""    
        # Transform string fields to actual data structure    
        def convert_fields(example):  
            # Transform all JSON string fields    
            for field in ['history']:  
                if isinstance(example[field], str):  
                    try:  
                        example[field] = json.loads(example[field].replace("'", '"'))  
                    except:  
                        example[field] = []  # Handle empty value cases    
            
            # for k, v in example.items():
            #     if isinstance(v, str):  
            #         try:  
            #             example[k] = json.loads(v.replace("'", '"'))  
            #         except:  
            #             example[k] = []  # Process空Value情况    
            return example 
        
        self.dataset = self.dataset.map(convert_fields)  
        print("huggingface dataset, each field of each sample may be a json string, so we need to transform, currently, transformation successful")  
        print("type(self.dataset[0][field]) = ", type(self.dataset[0][field]) )
        print("self.dataset[0][field] = ", self.dataset[0][field] )
        
        print("============ Start initializing data library with local embedding model ===============")  
        print("dataset = ", self.dataset)   
        print("type(dataset) = ", type(self.dataset))
        
        print("dataset[0] = ", self.dataset[0])
        
        if not (isinstance(self.dataset, Dataset)) or field not in self.dataset[0]:  
            print("dataset.features = ", self.dataset.features)
            raise ValueError(f"Dataset format does not meet requirements, should contain dictionary with '{field}' field")   
        
        
        # Add type check    
        sample = self.dataset[0]  
        # if not isinstance(sample[field], (dict, list)):  
        #     raise TypeError(f"{field}字段应为Dictionary/Dict/ListClass型，ActualClass型为 {type(sample[field])}")     
                
        
        
        
        sample = self.dataset[5][field]
        print("==========================================")
        print(f"the fifth {field} field sample = ", sample)
        print("===============================================")
        
    
        batch_size = 100  
        for i in range(0, len(self.dataset), batch_size):  
            batch:List[str] = self.dataset[i:i+batch_size] if i+batch_size<=len(self.dataset) else self.dataset[i:len(self.dataset)]
            # print("batch = ", batch)
            print("type(batch) = ", type(batch))
            
            count = 0
            for item in batch[field]:
                print("item = ", item)
                count+=1
                if count==2:
                    break
                # raise ValueError("item stop")
            documents = [str(item) for item in batch]  
            metadatas = [{"source": "crosswoz"}] * len(documents)  
            ids = [str(idx) for idx in range(i, i+len(documents))]  
            
            self.collection.add(  
                documents=documents,  
                metadatas=metadatas,  
                ids=ids  
            )  
    
    
    def query_db(self, user_query:str, n_results=5)->List[str]:
        assert self.use_db, "The embedding database is not initialized."
        
        # Usage本地ModelGenerateQueryEmbedding    
        query_embedding = self.embedding_fn([user_query])[0]
        
        # corpus = self.parse_db()
        # ids = [f"id{i+1}" for i in range(len(corpus))]
        
        # self.collection.add(
        #     documents = corpus,
        #     # metadatas = [{"source": "my_source"}, {"source": "my_source"}],
        #     ids = ids
        # )
        
        results = self.collection.query(
            query_embeddings = [query_embedding],
            # query_texts= [user_query],
            n_results = n_results,
            # where = {"metadata_field": "is_equal_to_this"},
            # where_document = {"$contains": "search_string"}
            include=["documents", "distances"] 
        )
        
        # Add similarity threshold filter    
        filtered = [  
            doc for doc, dist in zip(results["documents"][0], results["distances"][0])  
            if 1 - dist > 0.7  # Convert cosine distance to similarity    
        ]  
        
        return filtered
        
    
    
    def chat(self):
        '''
        simple chat without RAG
        '''
        self.agent.chat()
    
    def rag_chat(self,):
        history = [("You are a very help AI assistant who can help me plan a wonderful trip for a vacation",
                    "OK, I know you want to have a good travel plan and I will answer your questions about the traveling spot and search for the best plan about the traveling route and hotel.")]

        print("\n\n\n=============================")
        print("============ Welcome to the TravelMind Chat! Type 'exit' to stop chatting. ==========")  
        while True:  
            user_input = input(f"User: ")  
            if user_input.lower() == 'exit':  
                print("Goodbye!")  
                break  
            
            prompt = self.prompt_template.generate_prompt(
                user_input,
                "\n".join([f"User:{user}\nSystem:{sys}" for user, sys in history])
                )
            
            # formatted_history = " ".join([f"User: {user}\nSystem: {sys}\n" for user, sys in history])

            
            tool_call_str = self.agent.generate_response(
                prompt,
                max_length=2048
                )  
            
            print(" ================ ModelReturn/Back的Package含Tool调用的response =======================")  
            print(tool_call_str)
            print("===========================================")
            
            # Tool调用  
            raw_result = self.dispatcher.execute(tool_call_str)
            
            # DataLibraryMatch
            db_result = self.query_db(user_input) if self.use_db else ""
            db_result = "\n".join(db_result)
            
            final_response = tool_call_str + f"""
            
            Tool调用ResultYes/Is：  
            {raw_result}
            
            DataLibraryQuery的ResultYes/Is：  
            {db_result}
            """
            print("=============== 集成所有的ToolInformation后的prompt ===============")  
            print(final_response)
            print("=====================================================")
            
            travel_plan = self.get_travel_plan(final_response, max_length=256)
            # summary = self.summarize_results(final_response)
            # Summary
            print(f"TravelMind: {travel_plan}")  
            print(" ======================================= ")
            
            history.append((user_input, travel_plan))
    
    
    
    def langchain_rag_chat(self):
        """完全基于LangChain APIImplementation的Enhancement版对话"""    
        print("============ LangChain RAG Chat Start/Launch ===========")  
        
        
        # InitializeLangChainComponent  
        memory = ConversationBufferMemory(  
            return_messages=True,   
            output_key="answer",  
            memory_key="chat_history"  
        )  
        
        # Create检索器（Usage已Initialize的ChromaSet）    
        retriever = Chroma(  
            client=self.chroma_client,  
            collection_name="my_collection",  
            embedding_function=self.embeddings  # 需补充ActualembeddingModel    
        ).as_retriever(search_kwargs={"k": 5})   # Set/Configure每次检索Return/Back最相关的5个Result  
        
        # BuildTool调用Chain    
        tool_chain = (  
            RunnablePassthrough.assign(  
                context=lambda x: retriever.get_relevant_documents(x["question"])  
            )  
            | self._build_tool_prompt()  
            | self.agent.model  # False设已适配LangChainInterface    
            | StrOutputParser()  
        )  
        
        
        # Start/Launch对话Loop/Cycle    
        while True:  
            user_input = input("User: ")  
            if user_input.lower() == "exit":  
                print("Goodbye!")  
                break  
            
            response = tool_chain.invoke({  
                "question": user_input,  
                "chat_history": memory.load_memory_variables({})["chat_history"]  
            })  
            
            # ParseTool调用    
            tool_result = self._process_langchain_response(response)  
            memory.save_context({"input": user_input}, {"output": tool_result})  
            
            print(f"Assistant: {tool_result}")  
            print("=============================================")  
    
    
    def _build_tool_prompt(self):  
        """Build集成Tool和context的TooltipTemplate"""   
        
        return ChatPromptTemplate.from_template("""  
            结合以Down/BelowContext和Tool调用Result回答Question/Problem：    
            
            Context：  
            {context}  
            
            History对话：    
            {chat_history}  
            
            可调用的Tool：  
            {tools}
            
            UserQuestion/Problem：{question}  
            
            请按以Down/BelowFormatResponse：    
            {tool_format}  
            """).partial(  
                tool_format=self.prompt_template.get_tool_format(),
                tools = self.prompt_template.get_tools(),
            )  
            
    def _process_langchain_response(self, response: str) -> str:  
        """ProcessLangChainOutput并ExecuteTool调用"""    
        try:  
            # AddTool调用频率Restriction/Limitation    
            if len(re.findall(r"<Tool调用>", response)) > 10:    
                return "检测到过多Tool调用，请简化您的Question/Problem"  
            
            # ParseTool调用字符串    
            tool_call = re.search(r"<Tool调用>(.*?)</Tool调用>", response, re.DOTALL)    
            if not tool_call:  
                return response  

            # ExecuteTool调用    
            result = self.dispatcher.execute(tool_call.group(1).strip())  
             
            
            return f"{response}\n\nToolExecuteResult：{result}"  
        
        except Exception as e:  
            return f"Error processing response: {str(e)}"  
    
    
    def summarize_results(self, results:Dict)->str:
        """将原始ResultTransform为NatureLanguageDigest"""    
        summaries = []  
        for item in results.get("items", []):  
            summaries.append(f"标题：{item['title']}\nDigest：{item['snippet']}")    
        return "\n\n".join(summaries) 
    
    
    def get_travel_plan(self, query:str, max_length = 512):
        SYS_PROMPT = "你Yes/Is一个Travel/Trip助手，可以Help我规划一条Suitable/Appropriate的Travel/TourismRoute. 基于Down/Below面的Information: {query}, 请你帮我规划一条合理的Route/RoutingRoute. 你Return/Back的Route比如用List的形式组织，并且Clear，简洁."  

        
        response = self.agent.generate_response(SYS_PROMPT, max_length=max_length)
        
        
        return response
    
    


class CityRAG(RAG):
    '''
    基于 BM25 Match算法和城市Travel/TourismKnowledgeLibraryImplementation的RAG  
    '''
    def __init__(
        self, 
        agent: TravelMind = None,
        dataset_name_or_path:str = RAG_DATA_PATH,
        embedding_model_name_or_path:str = EMBEDDING_MODEL_PATH,
        use_langchain = False,
        use_prompt_template = True,
        use_db = True,
        use_api=True,
        folder_path=PAGE_FOLDER_PATH,
    ):
        
        super().__init__(
            agent=agent,
            dataset_name_or_path = dataset_name_or_path,
            embedding_model_name_or_path= embedding_model_name_or_path,
            use_langchain = use_langchain,
            use_prompt_template = use_prompt_template,
            use_db = use_db,
            use_api = use_api
        )
        print("BasicRAGObject构造完毕~~~")  
        self.corpus:Dict[str, List] = self.load_city_data(folder_path)
        
        print("城市Travel/TourismDataLoad完毕~~~")  
        
        self.bm25_model = BM25(self.corpus)
        
        
        
        
        
    def load_city_data(self, folder_path)->Dict[str, List[str]]:
        self.city_data = {}
        
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                    plan = file.read()
                    city = file_name.split(".")[0]
                    self.city_data[city] = plan
                    
        corpus = {}
        self.index_to_name = {}
        index = 0
        
        for city, plan in self.city_data.items():
            corpus[index] = jieba.lcut(plan)
            self.index_to_name[index] = city
            index+=1
            
        return corpus
    
    
    
    
    def retrive(self, user_query):
        scores = self.bm25_model.get_scores(jieba.lcut(user_query))
        
        sorted_scores = sorted(scores, key=lambda x:x[1], reverse = True)
        city_index = sorted_scores[0][0]
        
        text = self.city_data[self.index_to_name[city_index]]
        
        return text
    
    
    
    def query(self, user_query):
        print("user_query:", user_query)
        print("=======================")
        retrive_text = self.retrive(user_query)
        print("retrive_text:", retrive_text)
        print("=======================")
        prompt = f"请根据以Down/Below从DataLibraryMedium获得的Travel/TripRoute规划，回答UserQuestion/Problem：\n\n所有城市的Travel/Trip笔记，Attraction/Scenic spot、Cuisine/Delicacy、HotelRecommended：\n{retrive_text}\n\nUserQuestion/Problem：{user_query}"  
        response_text = self.call_api_model(prompt)
        print("Model回答：", response_text)  
        print("=======================")
        
        
        


    
class GraphRAG(RAG):
    def __init__(
        self, 
        agent: TravelMind,
        dataset_name_or_path:str = RAG_DATA_PATH,
        embedding_model_name_or_path:str = EMBEDDING_MODEL_PATH,
        use_langchain = False,
        use_prompt_template = True,
        use_db = True,
        use_api = False,
    ):
        super().__init__(
            agent=agent,
            dataset_name_or_path = dataset_name_or_path,
            embedding_model_name_or_path= embedding_model_name_or_path,
            use_langchain = use_langchain,
            use_prompt_template = use_prompt_template,
            use_db = use_db,
            use_api=use_api
        )
    
    def _initialize_knowledge_graph(self) -> Neo4jGraph:  
        """BuildKnowledge图谱"""    
        # Join到Neo4j（ExampleConfigure，需根据ActualModify）    
        graph = Neo4jGraph(  
            url="bolt://localhost:7687",  
            username="neo4j",  
            password="password"  
        )  
        
        # 从DocumentationMediumExtractEntity关系    
        query = """  
        UNWIND $documents AS doc  
        CALL apoc.nlp.gcp.entities.analyze({  
            text: doc.text,  
            key: $apiKey,  
            types: ["PERSON","LOCATION","ORGANIZATION"]  
        }) YIELD value  
        UNWIND value.entities AS entity  
        MERGE (e:Entity {name: entity.name})  
        SET e.type = entity.type  
        WITH e, doc  
        MERGE (d:Document {id: doc.id})  
        MERGE (d)-[:CONTAINS]->(e)  
        """  
        
        # BatchProcessDocumentation（Example）  
        documents = [{"id": str(i), "text": d["history"]} for i, d in enumerate(self.dataset)]  
        graph.query(query, params={"documents": documents, "apiKey": "your-gcp-key"})  
        
        return graph  
    
    
    
    def _build_graph_prompt(self) -> PromptTemplate:  
        """Build图谱Enhancement的TooltipTemplate"""    
        return PromptTemplate.from_template("""  
            结合Knowledge图谱和TextContext回答Down/Below列Question/Problem：    
            
            Knowledge图谱Path：    
            {graph_paths}  
            
            相关Text：    
            {context}  
            
            History对话：    
            {chat_history}  
            
            Question/Problem：{question}  
            
            请按照以Down/BelowRequirement回答：    
            1. Explicit/Clear提及相关Entity    
            2. Instruction/DescriptionEntity间的关系    
            3. Keep/Maintain回答简洁Professional    
            """)  
        
        
    
    def graph_rag_chat(self):  
        """基于GraphRAG的对话Implementation"""    
        from langchain.memory import ConversationBufferMemory  
        from langchain.chains import ConversationalRetrievalChain  
        
        # InitializeComponent  
        memory = ConversationBufferMemory(  
            memory_key="chat_history",  
            return_messages=True,  
            output_key="answer"  
        )  
        
        # Create混合检索器    
        class GraphEnhancedRetriever:  
            def __init__(self, vector_retriever, graph):  
                self.vector_retriever = vector_retriever  
                self.graph = graph  
                
            def get_relevant_documents(self, query: str) -> List[Dict]:  
                # Vector检索    
                vector_docs = self.vector_retriever.get_relevant_documents(query)  
                
                # 图谱检索    
                graph_query = f"""  
                MATCH path=(e1)-[r]->(e2)  
                WHERE e1.name CONTAINS '{query}' OR e2.name CONTAINS '{query}'  
                RETURN path LIMIT 5  
                """  
                graph_paths = self.graph.query(graph_query)  
                
                return {  
                    "vector_docs": vector_docs,  
                    "graph_paths": graph_paths  
                }  

        # Initialize检索器    
        vector_retriever = Chroma(  
            client=self.chroma_client,  
            collection_name="my_collection"  
        ).as_retriever()  
        
        hybrid_retriever = GraphEnhancedRetriever(vector_retriever, self.graph)  
        
        # Create对话Chain    
        qa_chain = ConversationalRetrievalChain.from_llm(  
            llm=self.agent.llm,  
            retriever=hybrid_retriever,  
            memory=memory,  
            combine_docs_chain_kwargs={  
                "prompt": self._build_graph_prompt(),  
                "document_prompt": PromptTemplate(  
                    input_variables=["page_content"],  
                    template="{page_content}"  
                )  
            },  
            get_chat_history=lambda h: "\n".join([f"User:{u}\nAssistant:{a}" for u, a in h])  
        )  
        
        # Start/Launch对话Loop/Cycle    
        print("========== GraphRAG对话SystemStart/Launch ==========")    
        while True:  
            try:  
                query = input("User: ")  
                if query.lower() in ["exit", "quit"]:  
                    break  
                
                result = qa_chain({"question": query})  
                print(f"助手: {result['answer']}")    
                print("\nKnowledge图谱Path:")    
                for path in result["graph_paths"]:  
                    print(f"- {path['start_node']['name']} → {path['relationship']} → {path['end_node']['name']}")  
                print("=====================================")  
                
            except KeyboardInterrupt:  
                break  







class RagDispatcher:
    def __init__(
        self,
        rag_type:str = "rag"
    ):
        pass
    



    
# HelperFunction 
def visualize_knowledge_graph(graph: Neo4jGraph):  
    """VisualizationKnowledge图谱（Example）"""    
    query = """  
    MATCH (n)-[r]->(m)  
    RETURN n.name AS source,   
           type(r) AS relationship,   
           m.name AS target  
    LIMIT 50  
    """  
    return graph.query(query)  
        
        
        
        
if __name__ == '__main__':
    pass
    
        
    
        