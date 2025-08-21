
from src.agents.prompt_template import MyPromptTemplate
from src.agents.tools import ToolDispatcher
from typing import Dict, List, Optional, Tuple
# from src.models.model import TravelMind
from src.data.data_processor import CrossWOZProcessor



import langchain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import ChatVectorDBChain
from langchain.chains.llm import LLMChain

from langchain.memory.buffer import ConversationBufferMemory
from langchain_community.vectorstores.chroma import Chroma      # pip install langchain-chroma  pip install langchain_community
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

from langchain_community.llms.tongyi import Tongyi
from langchain_community.llms.openai import OpenAI
from langchain_community.embeddings.openai import OpenAIEmbeddings

from langchain_community.document_loaders.pdf import PyPDFLoader



from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin



from typing import Dict, List, Optional, Tuple, Literal


'''
Recommend checking Pydantic version compatibility. Recommended usage:
pip install pydantic>=2.5.0

'''

from datasets import load_dataset
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
import re
import os
import torch
from zhipuai import ZhipuAI

from src.configs.config import RAG_DATA_PATH, SFT_MODEL_PATH, EMBEDDING_MODEL_PATH


ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")

class MyAgent():
    """
    A specialized travel planning optimization agent that contains several sub-agents:
    1. Structure Analysis Agent: Analyzes document structure and topics, provides optimization suggestions.
    2. Language Optimization Agent: Checks grammar errors and inappropriate word usage, provides optimization suggestions.
    3. Content Enrichment Agent: Based on document topics, proposes further extensions and enrichment points or improvement suggestions.
    """

    def __init__(
        self,
        use_api:bool = True,
        travel_agent=None,
        use_rag = False,
        use_langchain_agent = False
        ):
        self.use_api = use_api
        self.use_rag = use_rag
        self.use_langchain_agent = use_langchain_agent


        if not self.use_api:
            self.agent = travel_agent


        self.base_template = "You are a travel planning assistant with 10 years of tour guide experience. You are very familiar with overseas travel business including but not limited to: flight tickets, hotels, cuisine, transportation, attractions, and local guides."
    def call_local_model(self, prompt):

        # Load local model        model:AutoModelForCausalLM|GenerationMixin = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH)
        model = model.to("cuda")

        tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)

        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=512)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response



    # https://open.bigmodel.cn/ Register and obtain API Key
    # https://www.bigmodel.cn/dev/api/normal-model/glm-4  API Documentation
    def call_api_model(self, prompt)->str:
        client = ZhipuAI(api_key=ZHIPU_API_KEY)  # Fill in your own API Key        response = client.chat.completions.create(
            model="glm-4-flash",  # Fill in the model name to call            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        response_text = response.choices[0].message.content
        return response_text


    def plan_structure_agent(self, travel_plan_response:str):
        """Query the large model for travel plan analysis, return the structure of the current travel plan"""        prompt_analysis = self.base_template + f"Please organize, analyze and output the structure of the following travel plan, including key nodes, key routes and transportation methods, key operations: {travel_plan_response}"        # Call large model interface, the returned result is a dictionary containing structure and topic information
        if self.use_api:
            plan_structure_result = self.call_api_model(prompt_analysis)
        else:
            plan_structure_result = self.call_local_model(prompt_analysis)

        return plan_structure_result


    def language_optimization_agent(self, travel_plan_response, plan_structure_result):
        # Build prompts based on route planning structure        prompt_language = self.base_template + f"Please check the grammar errors and inappropriate word usage in the following travel route planning, and provide optimization suggestions. Suggestions should be concise, no more than 200 words.\n\nTravel planning structure: {plan_structure_result}\n\nTravel planning content: {travel_plan_response}"        language_optimization_suggestions = self.call_api_model(prompt_language)
        return language_optimization_suggestions

    def content_enrichment_agent(self, travel_plan_response, plan_structure_result):
        # Build prompts based on article analysis results        prompt_content = self.base_template + f"Please read the following travel planning solution, based on the provided travel planning structure, propose content points that can be further extended and enriched or improvement suggestions, such as adding nearby recommendations, correcting erroneous data, updating shortest paths, etc. Suggestions should be concise, no more than 100 words.\n\nTravel planning structure: {plan_structure_result}\n\nTravel planning content: {travel_plan_response}"        content_enrichment_suggestions = self.call_api_model(prompt_content)
        return content_enrichment_suggestions

    def readability_evaluation_agent(self, travel_plan_response, plan_structure_result):
        # Build prompts based on article analysis results        prompt_readability = self.base_template + f"Please read the following travel planning solution, evaluate the readability of the plan based on the provided travel planning structure, including paragraph length, sentence complexity, etc., and provide improvement suggestions that help with plan implementation. Suggestions should be concise, no more than 100 words.\n\nTravel planning structure: {plan_structure_result}\n\nTravel planning content: {travel_plan_response}"        readability_evaluation_result = self.call_api_model(prompt_readability)
        return readability_evaluation_result

    def comprehensive_optimization_agent(self, travel_plan_response, plan_structure_result, language_optimization_suggestions, content_enrichment_suggestions, readability_evaluation_result):
        # The logic of merging results is to organize suggestions from each part into structured documentation        final_optimization_plan = self.base_template + f"Please read the following travel route planning and the improvement suggestions given by several specialized optimization agents, revise this travel planning solution to improve the overall quality of the planning content.\n\nOriginal travel planning: {travel_plan_response}\n\nTravel planning structure: {plan_structure_result}\n\nLanguage optimization suggestions: {language_optimization_suggestions}\n\nContent enrichment suggestions: {content_enrichment_suggestions}\n\nReadability improvement suggestions: {readability_evaluation_result}.\n\nOptimized travel planning solution:"        final_optimization_result = self.call_api_model(final_optimization_plan)
        return final_optimization_result



    def get_final_plan(self, travel_plan_response):
        '''
        Use Chain to sequentially call multiple agents to complete the task
        '''
        structure = self.plan_structure_agent(travel_plan_response)
        language_optimization_suggestions = self.language_optimization_agent(travel_plan_response, structure)
        content_enrichment_suggestions = self.content_enrichment_agent(travel_plan_response, structure)
        readability_evaluation_result = self.readability_evaluation_agent(travel_plan_response, structure)
        final_result = self.comprehensive_optimization_agent(travel_plan_response, structure, language_optimization_suggestions, content_enrichment_suggestions, readability_evaluation_result)

        return final_result






class AgentWithLangChain():
    def __init__(
        self,
        chain_type:Literal["stuff", "map_reduce", "refine"] = "stuff",
        ):
        pass

if __name__ == '__main__':
    pass
