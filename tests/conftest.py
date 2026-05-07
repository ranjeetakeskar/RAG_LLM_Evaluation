import pytest
from langchain_openai import ChatOpenAI
from ragas.llms import  LangchainLLMWrapper
from dotenv import load_dotenv



def pytest_configure(config):
    load_dotenv()

@pytest.fixture(scope="function")
def init_llm_wrapper(request):
    llm= ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0)
    ragas_llm_wrapper=LangchainLLMWrapper(llm)
    yield ragas_llm_wrapper
    
    