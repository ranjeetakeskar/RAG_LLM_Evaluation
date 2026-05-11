import os

from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from dotenv import load_dotenv
import nltk
import asyncio
import sys

from ragas.testset.persona import Persona

load_dotenv()

# Windows async stability fix
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

# Reduce ragas concurrency
os.environ["RAGAS_MAX_WORKERS"] = "1"

def test_data_generation():
    nltk.data.path.append(os.getcwd()+"/nltk_data/")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    langchain_llm = LangchainLLMWrapper(llm)
    embed = OpenAIEmbeddings()
    file_path = os.getcwd() + f"/reference_docs/"
    loader = DirectoryLoader(
        path=file_path,
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader
    )
    docs = loader.load()
    
    generate_embeddings = LangchainEmbeddingsWrapper(embed)
    generator = TestsetGenerator(llm=langchain_llm, embedding_model=generate_embeddings)
    data_set=generator.generate_with_langchain_docs(docs, testset_size=2)
    print(f"Data set {data_set}")

    
    
    
    
    

# def test_openai_direct():
  
   
#     llm = ChatOpenAI(
#         model="gpt-4o-mini",
#         temperature=0
#     )

#     response = llm.invoke("Hello")
#     print(response.content)