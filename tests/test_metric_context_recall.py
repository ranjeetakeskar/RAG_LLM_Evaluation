import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics._context_recall import LLMContextRecall
from utilities import  util

def test_context_recall():
    load_dotenv()
    #1. Init open api
    llm= ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0)
    ragas_llm_wrapper=LangchainLLMWrapper(llm)
    # 2. init LLM context recall
    context_recall=LLMContextRecall(llm=ragas_llm_wrapper)
    # 3. Feed data, for single query
    responseData=Util.get_llm_response()
    retrieved_docs_data=[]
    for i in range(len(responseData["retrieved_docs"])):
        retrieved_docs_data.append(responseData["retrieved_docs"][i]["page_content"])

    sample= SingleTurnSample(
        user_input="How many articles are there in the Selenium webdriver python course",
        retrieved_contexts= retrieved_docs_data,
        reference= "23"

    )

    context_recall_score=  context_recall.single_turn_score(sample)
    print(f"context recall score: {context_recall_score}")
    assert context_recall_score > 0.7

