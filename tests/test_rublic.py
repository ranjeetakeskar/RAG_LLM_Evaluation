import pytest
from ragas import SingleTurnSample
from ragas.metrics._domain_specific_rubrics import RubricsScore

@pytest.fixture
def get_data():
    sample = SingleTurnSample(
        user_input="Where is Effiel Tower located?",
        response="Effiel Tower is located in Europe and it is part of France",
        reference="Effiel Tower is located in paris"
    )

    return sample


def test_rubrics_score(init_llm_wrapper, get_data):
    rubrics = {
        "score1_description": "The response is incorrect, irrelevant, or does not align with the ground truth.",
        "score2_description": "The response partially matches the ground truth but includes significant errors, omissions, or irrelevant information.",
        "score3_description": "The response generally aligns with the ground truth but may lack detail, clarity, or have minor inaccuracies.",
        "score4_description": "The response is mostly accurate and aligns well with the ground truth, with only minor issues or missing details.",
        "score5_description": "The response is fully accurate, aligns completely with the ground truth, and is clear and detailed.",
    }

    rubics_score=RubricsScore(rubrics=rubrics, llm=init_llm_wrapper)
    score=rubics_score.single_turn_score(get_data)
    print(f"Rubics score:  {score}")
