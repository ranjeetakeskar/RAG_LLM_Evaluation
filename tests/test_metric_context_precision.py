import os
import pytest
from ragas import SingleTurnSample
from ragas.metrics._context_precision  import LLMContextPrecisionWithoutReference
from langchain_openai import ChatOpenAI
from ragas.llms import  LangchainLLMWrapper

from utilities import util
from dotenv import  load_dotenv

@pytest.fixture
def get_data(request):
    data = request.param
    responseData= util.get_llm_response(data["question"])
    retrieved_docs_data=[]
    for i in range(len(responseData["retrieved_docs"])):
        retrieved_docs_data.append(responseData["retrieved_docs"][i]["page_content"])

    sample = SingleTurnSample(
        user_input= data["question"],
        response=responseData["answer"],
        retrieved_contexts=retrieved_docs_data
    )
    return sample


@pytest.mark.parametrize("get_data",util.load_test_data("context_precision.json"),indirect=True)
def test_context_precision(init_llm_wrapper, get_data):
    # create object of precision class
    context_precision = LLMContextPrecisionWithoutReference(llm=init_llm_wrapper)
    # get score for single query
    context_precision_score = context_precision.single_turn_score(get_data)
    print(f"context precision score: {context_precision_score}")
    assert context_precision_score > 0.8






    
