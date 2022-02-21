from fastapi import FastAPI
import numpy as np
from typing import List
from pydantic import BaseModel

from predict import init_global_obj, predict_f

"""
TODO
- [x] implement the BaseClass object for data validation
"""

class RequestBody(BaseModel):
    """
    is this the right way to do this?
    """
    product_list: List[str]

LANG = 'en'

label_classes, model, vectorizer = init_global_obj(LANG)


app = FastAPI()

@app.get('/')
def home():
    return {'ciao': 'Welcome to the API prediction'}

@app.post('/predict')
def predict(lang: str, request: RequestBody):
    json_response = {}
    product_list = request.product_list
    input_text = vectorizer(np.array([[product] for product in product_list])).numpy()
    prediction_list, confidence_scores = predict_f(model, input_text, label_classes)
    for product, prediction, confidence_score in zip(product_list, prediction_list, confidence_scores):
        json_response[product] = {"prediction":prediction, "confidence":round(confidence_score.item(), 4)}
    return json_response