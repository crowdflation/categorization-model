from enum import Enum
from fastapi import FastAPI
import numpy as np
from typing import List
from pydantic import BaseModel

from scripts.predict import init_global_obj, predict_f


class RequestBody(BaseModel):
    product_list: List[str]

class SupportedLangs(str, Enum):
    en = "en"
    tr = "tr"

lang_classes = {}
models = {}
vectorizers = {}

for lang in SupportedLangs:
    label_classes, model, vectorizer = init_global_obj(lang.value)
    lang_classes[lang.value] = label_classes
    models[lang.value] = model
    vectorizers[lang.value] = vectorizer

api = FastAPI()

@api.get('/')
def home():
    return {'home': 'Welcome to the API for categorization. We use AI models to classify products&services names across a variety of categories.'}

@api.post('/predict/{lang}')
def predict(lang: SupportedLangs, request: RequestBody):
    json_response = {}
    product_list = request.product_list
    input_text = vectorizers[lang.value](np.array([[product] for product in product_list])).numpy()
    prediction_list, confidence_scores = predict_f(models[lang.value], input_text, lang_classes[lang.value])
    for product, prediction, confidence_score in zip(product_list, prediction_list, confidence_scores):
        json_response[product] = {"prediction":prediction, "confidence":round(confidence_score.item(), 4)}
    return json_response