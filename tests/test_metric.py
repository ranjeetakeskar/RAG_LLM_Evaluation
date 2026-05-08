# test for response relavancy and factual correctness
import pytest
from ragas.metrics._answer_relevance import ResponseRelevancy
from ragas.metrics._factual_correctness import FactualCorrectness
from utilities import util
from ragas import SingleTurnSample, EvaluationDataset, evaluate
import pytest

@pytest.fixture
def get_data(request):
    data=request.param
    responseData=util.get_llm_response(data["question"])
    retrieved_docs_data=[]
    for i in range(len(responseData["retrieved_docs"])):
        retrieved_docs_data.append(responseData["retrieved_docs"][i]["page_content"])

    sample= SingleTurnSample (
        user_input= data["question"],
        reference= data["reference"],
        response= responseData["answer"],
        retrieved_contexts= retrieved_docs_data
    )

    return sample

@pytest.mark.parametrize("get_data",util.load_test_data("response_factual.json"), indirect=True)
def test_metric_relavancy_factual(init_llm_wrapper,get_data):
    metrics= [ ResponseRelevancy(llm= init_llm_wrapper), FactualCorrectness(llm=init_llm_wrapper)]
    eval_data=EvaluationDataset([get_data])
    results=evaluate(dataset=eval_data, metrics=metrics)
    print(results)
    print(results["answer_relevancy"])
    



@pytest.mark.parametrize("get_data",util.load_test_data("response_factual.json"), indirect=True)
def test_metric_standard(init_llm_wrapper,get_data):

    # when no metric is given, then ragas will generate standard metric
    eval_data=EvaluationDataset([get_data])
    results=evaluate(dataset=eval_data)
    print(results)
    

   