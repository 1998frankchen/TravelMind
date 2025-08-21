
from typing import List, Optional, Dict, Literal, Tuple, Any
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.llm import LLMChain



from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.vectorstores.utils import filter_complex_metadata



from langchain_core.runnables import RunnableLambda, RunnableParallel

from zhipuai import ZhipuAI
import os

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, pipeline

from src.configs.config import MODEL_PATH,EMBEDDING_MODEL_PATH_BPE, SFT_MODEL_PATH, EMBEDDING_MODEL_PATH, PDF_FOLDER_PATH
from src.agents.chat_pdf import ChatPDF

import asyncio





class SelfRAGBase:

    def __init__(self):
        pass
    async def _call_model(self, prompt: ChatPromptTemplate, inputs: Dict, model_type: Literal["api", "huggingface"]) -> str:
        if model_type == "api":
            client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
            response = client.chat.completions.create(
                model="glm-4",
                messages=[{"role": "user", "content": prompt.format(**inputs)}]
            )
            return response.choices[0].message.content
        elif model_type == "huggingface":
            tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH, trust_remote_code = True)
            model = AutoModelForSeq2SeqLM
            pipe = pipeline(
                task="text-generation",
                model = model,
                tokenizer=tokenizer,
                max_new_tokens = 200,
            )
            model = HuggingFacePipeline(pipeline=pipe)
            chain = LLMChain(llm=model, prompt=prompt)
            return await chain.arun(inputs)
        else:
            raise ValueError("Invalid model_type, please choose either 'api' or 'huggingface'")


    # def _init_vector_store(self):
    #     """Initialize local PDF document library"""
    #     if not self._vector_store:
    #         # Load local PDF documents
    #         loader = PyPDFLoader("knowledge_base.pdf")
    #         documents = loader.load()

    #         # Document chunking
    #         text_splitter = RecursiveCharacterTextSplitter(
    #             chunk_size=500,
    #             chunk_overlap=50
    #         )
    #         splits = text_splitter.split_documents(documents)

    #         # Create vector storage
    #         embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh")
    #         self._vector_store = Chroma.from_documents(
    #             documents=splits,
    #             embedding=embeddings
    #         )





class ChatPDFForSelfRAG(ChatPDF):
    def __init__(self, pdf_folder_path:str = PDF_FOLDER_PATH, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_chunks:List[Document] = []  # Storage分块Result
        self.vector_store = None
        self.retriever=None

        self.pdf_folder_path = pdf_folder_path

        self.ingest_all(self.pdf_folder_path)
        print("ChatPDF vector library initialization completed")

    def ingest_all(self, pdf_folder_path: str):
        """Override parent class method to store chunking results"""

        assert self.vector_store is None, "Vector store has already been initialized. Do not ingest again."

        self.all_chunks = []
        for file_name in os.listdir(pdf_folder_path):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(pdf_folder_path, file_name)
                docs = PyPDFLoader(file_path=file_path).load()
                chunks = self.text_spliter.split_documents(docs)
                chunks = filter_complex_metadata(chunks)
                self.all_chunks.extend(chunks)

        # Initialize vector storage (maintain original functionality)
        self.vector_store = Chroma.from_documents(
            documents=self.all_chunks,
            # embedding=FastEmbedEmbeddings(model_name = "BAAI/bge-small-en-v1.5")   # Note: this doesn't support local paths
            embedding=HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_PATH,
                model_kwargs={'device': 'cpu'},  # or 'cpu'
                encode_kwargs={'normalize_embeddings': True}

                )
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.5}
        )

    def get_selfrag_chunks(self) -> List[Document]:
        """Fetch document chunks suitable for Self-RAG"""
        return [
            Document(
                page_content=chunk.page_content,
                metadata={"source": chunk.metadata.get("source", "")}
            ) for chunk in self.all_chunks
        ]


class SelfRAG(SelfRAGBase):
    def __init__(self, model_type="api"):
        self.model_type = model_type
        self.retrieve_threshold = 0.2
        self.beam_width = 2

        self.chat_pdf = ChatPDFForSelfRAG(pdf_folder_path=PDF_FOLDER_PATH)


        # Define reflection tag prompt templates
        self.retrieve_prompt = ChatPromptTemplate.from_template(
            "Do you need to retrieve documents to answer the following question? Answer Yes or No\nQuestion: {input}"
        )
        self.isrel_prompt = ChatPromptTemplate.from_template(
            "Evaluate/EvaluationQuestion/Problem和所给段落的相关性：\nQuestion/Problem：{input}\n段落：{context}\n相关吗？(Relevant/Irrelevant)"
        )
        self.issup_prompt = ChatPromptTemplate.from_template(
            "Verify/Validate所给段落对Generate内容Support程度：\nGenerate内容：{generation}\n段落：{context}\nSupport程度？(Fully/Partially/No)"
        )
        self.isuse_prompt = ChatPromptTemplate.from_template(
            "给出回答的质量Score/Rating（1-5）：\nQuestion/Problem：{input}\n回答：{generation}\nScore/Rating："
        )


    async def retrieve(self, query: str, k: int=5) -> List[str]:
        """
        检索
        """
        docs = self.chat_pdf.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    async def generate_segment(self, context: Optional[str],
                             input_str: str) -> str:
        """Generate单个段落"""
        prompt_template = ChatPromptTemplate.from_template(
            "Context：{context}\n\nGenerate回答：{input}" if context else "直接Generate回答：{input}"
        )
        return await self._call_model(
            prompt=prompt_template,
            inputs={"context": context, "input": input_str},
            model_type=self.model_type
        )

    async def critique(
        self,
        input_str: str,
        generation: str,
        context: Optional[str],
        aspect: str
        ) -> float:
        """CommentGenerate的内容"""
        prompts = {
            "ISREL": (self.isrel_prompt, ["input", "context"]),
            "ISSUP": (self.issup_prompt, ["generation", "context"]),
            "ISUSE": (self.isuse_prompt, ["input", "generation"])
        }

        prompt, input_keys = prompts[aspect]
        inputs = {"input": input_str, "generation": generation, "context": context}
        response = await self._call_model(
            prompt=prompt,
            inputs={k: inputs[k] for k in input_keys},
            model_type=self.model_type
        )
        return self._parse_critique(response, aspect)

    def _parse_critique(self, response: str, aspect: str) -> float:
        """ParseCommentResult"""
        # if aspect == "ISUSE":
        #     return int(response.strip()) / 5
        # return 1.0 if "Relevant" in response or "Fully" in response else 0.5

        response = response.strip().lower()

        try:
            if aspect == "ISUSE":
                # Extract数字Score/Rating（Support"Score/Rating：4"等Format）
                score_str = "".join(filter(str.isdigit, response))
                if not score_str:
                    return 0.6  # DefaultMedium等Score/Rating
                score = min(max(int(score_str), 1), 5)
                return score / 5

            elif aspect == "ISREL":
                if "irrelevant" in response:
                    return 0.0
                elif "relevant" in response:
                    return 1.0
                else:  # 无法Judge/Check时Medium等Score/Rating
                    return 0.5

            elif aspect == "ISSUP":
                if "no support" in response:
                    return 0.0
                elif "partially" in response:
                    return 0.5
                elif "fully" in response:
                    return 1.0
                else:  # Default部Branch持
                    return 0.5

        except Exception as e:
            print(f"ParseWrong/Error：{str(e)}，Return/BackDefaultValue")
            return 0.5 if aspect != "ISUSE" else 0.6

    async def build_chain(self):
        """BuildSelf-RAGChain"""
        return  (
            RunnableParallel({
                "input": lambda x: x,
                "retrieve_decision": await self._retrieve_decision_chain()
            })
            .pipe(self._generate_with_retrieval)
            .pipe(self._critique_and_select)
        )

    async def _retrieve_decision_chain(self):
        """检索DecisionChain"""
        return (
            RunnableLambda(lambda x: x)
            | StrOutputParser()
            | RunnableLambda(self._should_retrieve)
        )

    async def _should_retrieve(self, text: str) -> dict:
        """Judge/CheckYes/IsNo/Not需要检索"""
        decision = await self._call_model(
            prompt=self.retrieve_prompt,
            inputs={"input": text},
            model_type=self.model_type
        )
        return {"retrieve": "Yes" in decision, "text": text}

    async def _generate_with_retrieval(self, data: dict) -> dict:
        """带检索的Generate"""
        if data["retrieve_decision"]["retrieve"]:
            contexts = await self.retrieve(data["input"], k=3)
            return {"contexts": contexts, "input": data["input"]}
        return {"contexts": [], "input": data["input"]}

    async def _critique_and_select(self, data: dict) -> str:
        """Evaluate/Evaluation并Select最佳Result"""
        candidates = []
        tasks = []

        # ParallelismProcess候选段落
        for context in data.get("contexts", [])[:self.beam_width]:
            tasks.append(self._process_candidate(context, data["input"]))

        if not tasks:
            return await self.generate_segment(None, data["input"])

        candidates = await asyncio.gather(*tasks)
        return max(candidates, key=lambda x: x["score"])["text"]




    async def _process_candidate(self, context: str, input_str: str) -> dict:
        """Process单个候选"""
        generation = await self.generate_segment(context, input_str)

        # ParallelismEvaluate/Evaluation多个Dimension
        scores = await asyncio.gather(
            self.critique(input_str, generation, context, "ISREL"),
            self.critique(input_str, generation, context, "ISSUP"),
            self.critique(input_str, generation, None, "ISUSE")
        )

        return {
            "text": generation,
            "score": 0.4*scores[0] + 0.4*scores[1] + 0.2*scores[2]
        }


async def main():
    rag = SelfRAG(model_type="api")
    chain = await rag.build_chain()

    question = "请你帮我规划一个3天Up/Above海玩的Itinerary。"
    result = await chain.ainvoke(question)
    print(f"最终Answer：{result}")

if __name__ == "__main__":
    asyncio.run(main())