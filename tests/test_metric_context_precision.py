import os
import pytest
from ragas import SingleTurnSample
from ragas.metrics._context_precision  import LLMContextPrecisionWithoutReference
from langchain_openai import ChatOpenAI
from ragas.llms import  LangchainLLMWrapper

from utilities import util
from dotenv import  load_dotenv



def test_context_precision():
    # 1. create object of precision class
    load_dotenv()
    llm=ChatOpenAI(model="gpt-4.1-2025-04-14",temperature=0)
    ragas_llm_wrapper= LangchainLLMWrapper(llm)
    context_precision = LLMContextPrecisionWithoutReference(llm=ragas_llm_wrapper)
    # # # 2. Feed data , this for single query
    responseData= util.get_llm_response()
    retrieved_docs_data=[]
    for i in range(len(responseData["retrieved_docs"])):
        retrieved_docs_data.append(responseData["retrieved_docs"][i]["page_content"])

    sample = SingleTurnSample(
        user_input="How many articles are there in the Selenium webdriver python course",
        response=responseData["answer"],
        retrieved_contexts=retrieved_docs_data
    )
    # # 3. Score
    context_precision_score = context_precision.single_turn_score(sample)
    print(f"context precision score: {context_precision_score}")
    assert context_precision_score > 0.8






    
