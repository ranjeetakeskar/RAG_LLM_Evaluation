import pytest
from ragas.metrics._faithfulness import Faithfulness
from utilities import util
from ragas import SingleTurnSample

@pytest.fixture
def get_data(request):
    data=request.param

    responseData=util.get_llm_response(data["question"])
    retrieved_docs_data=[]
    for i in range(len(responseData["retrieved_docs"])):
        retrieved_docs_data.append(responseData["retrieved_docs"][i]["page_content"])

    sample = SingleTurnSample(
        user_input= data["question"],
        response=responseData["answer"],
        retrieved_contexts=retrieved_docs_data
    )
    return sample

@pytest.mark.parametrize("get_data",util.load_test_data("faithfulness.json"), indirect=True)
def test_faithfulness(init_llm_wrapper,get_data):
    #init llm
    faithfulness=Faithfulness(llm=init_llm_wrapper)
    #get score for single query
    faithfulness_score=  faithfulness.single_turn_score(get_data)
    print(f"context recall score: {faithfulness_score}")
    assert faithfulness_score > 0.7
