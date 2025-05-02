from src.agents.mem_walker import MemoryTreeNode
from src.agents.mem_walker import MemoryTreeBuilder

from src.agents.mem_walker import ChatPDFForMemWalker

from src.agents.mem_walker import Navigator

from src.agents.self_rag import SelfRAG


from typing import Literal, Callable, Dict, Tuple


from src.configs.config import PDF_FOLDER_PATH

import asyncio

# Input: Travel planning path  


class RAGDispatcher():
    """
    RAG (Retrieval-Augmented Generation) dispatcher that routes queries
    to different RAG implementations based on the specified type.

    Supports multiple RAG variants including self-RAG, corrective RAG,
    and memory walker for enhanced information retrieval and generation.
    """
    def __init__(self, rag_type:Literal["rag","self_rag", "corrective_rag", "mem_walker"]="mem_walker"):
        """
        Initialize the RAG dispatcher.

        Args:
            rag_type: The type of RAG implementation to use
        """
        self.rag_type = rag_type

    async def dispatch(self, query:str):
        """
        Dispatch the query to the appropriate RAG implementation.

        Args:
            query: The input query string

        Returns:
            The response from the selected RAG implementation
        """
        # 1. Planning path analysis
        # 2. Planning path execution
        # 3. Planning path summary  
        
        if self.rag_type == "mem_walker":
            return await self.mem_walker(query)
        
        elif self.rag_type == "rag":
            return self.rag(query)

        elif self.rag_type == "self_rag":
            return await self.self_rag(query)
        elif self.rag_type == "corrective_rag":
            return self.corrective_rag(query)
        
    def rag(self, query:str):
        """
        Basic RAG implementation (placeholder).

        Args:
            query: The input query string
        """
        pass
    
    async def mem_walker(self,query:str)->str:
        """
        Memory Walker RAG implementation that builds a hierarchical memory tree
        from PDF documents and navigates it to answer queries.

        Args:
            query: The input query string

        Returns:
            The generated answer from memory tree navigation
        """
        builder = MemoryTreeBuilder()
        
        pdf_reader = ChatPDFForMemWalker()
        pdf_reader.ingest_all(pdf_folder_path=PDF_FOLDER_PATH)
        
        all_chunks = pdf_reader.get_memwalker_chunks()
        root = await builder.build_tree(all_chunks, model_type="api")
        
        builder.print_memory_tree(root)
    
        navigator = Navigator(model_type="api")
        answer = await navigator.navigate(
            root, 
            query
            )
        
        
        return answer
        
        
    
    
    
    async def self_rag(self, query:str):
        """
        Self-RAG implementation that incorporates self-reflection mechanisms
        for improved retrieval and generation quality.

        Args:
            query: The input query string

        Returns:
            The generated response with self-reflection
        """
        rag = SelfRAG(model_type="api")  
        chain = await rag.build_chain()  
        
        result = await chain.ainvoke(query)  
        print(f"Final Answer: {result}")    
    
    
    
    def corrective_rag(self, query:str):
        """
        Corrective RAG implementation that includes mechanisms for
        correcting and refining retrieved information.

        Args:
            query: The input query string
        """
        pass

