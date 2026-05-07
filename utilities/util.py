import os

import requests


def get_llm_response():
    url_llm = "https://rahulshettyacademy.com/rag-llm/ask"
    data_json = {
        "question": "How many articles are there in the Selenium webdriver python course?",
        "chat_history": []
    }
    res = requests.post(url=url_llm, json=data_json)
    res.raise_for_status()
    return res.json()

