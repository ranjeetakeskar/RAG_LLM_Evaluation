import pytest
from ragas.metrics._topic_adherence import TopicAdherenceScore 
from ragas import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage



@pytest.fixture
def get_data():
    conversation= [
            HumanMessage(content="How many articles are there in the Selenium webdriver python course?"),
            AIMessage(content="There are 23 articles in the Selenium WebDriver Python course."),
            HumanMessage(content="how many are downloadable?"),
            AIMessage(content="There are 9 downloadable resources in the Selenium WebDriver Python course.")
        
    ]
    reference= ["""
                AI should give
                1. Give results related to selenium webdriver python course
                2. There are 23 articles and 9 downloadable resources in the Selenium WebDriver Python course."""]
    
    
    multi_conversation=MultiTurnSample(user_input=conversation, reference_topics=reference)
    return multi_conversation
    
    

def test_metric_multi_conversation(init_llm_wrapper, get_data):
    multi_conversation= TopicAdherenceScore(llm=init_llm_wrapper)
    multi_conversation_score=multi_conversation.multi_turn_score(get_data)
    print(f"Multi conversation score: {multi_conversation_score}")
    assert multi_conversation_score > 0.8