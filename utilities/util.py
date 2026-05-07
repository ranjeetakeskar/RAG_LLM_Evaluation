import os

import requests
import json

def get_llm_response(question):
    url_llm = "https://rahulshettyacademy.com/rag-llm/ask"
    data_json = {
        "question": question,
        "chat_history": []
    }
    res = requests.post(url=url_llm, json=data_json)
    res.raise_for_status()
    return res.json()



def load_test_data(filename):
    file_path= os.getcwd()+ f"/data/{filename}"
    if not os.path.exists(file_path):
        raise f"{filename} does not exists"

    with open(file_path) as file:
        data= json.load(file)
        return data




