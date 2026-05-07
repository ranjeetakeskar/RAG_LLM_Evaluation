import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics._context_recall import LLMContextRecall
from utilities import  util

@pytest.fixture
def get_data(request):
    data= request.param
    responseData=util.get_llm_response(data["question"])
    retrieved_docs_data=[]
    for i in range(len(responseData["retrieved_docs"])):
        retrieved_docs_data.append(responseData["retrieved_docs"][i]["page_content"])

    sample= SingleTurnSample(
        user_input=data["question"],
        retrieved_contexts= retrieved_docs_data,
        reference= data["reference"]

    )
    return sample


@pytest.mark.parametrize("get_data",util.load_test_data("context_recall.json"), indirect=True)
def test_context_recall(init_llm_wrapper, get_data):
    # init LLM context recall
    context_recall=LLMContextRecall(llm=init_llm_wrapper)
    # get score for single query
    context_recall_score=  context_recall.single_turn_score(get_data)
    print(f"context recall score: {context_recall_score}")
    assert context_recall_score > 0.7

