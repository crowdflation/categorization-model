from fastapi import FastAPI
import numpy as np

#from public_repo.predict import init_global_obj
from predict import init_global_obj, predict_f


LANG = 'en'

label_classes, model, vectorizer = init_global_obj(LANG)


app = FastAPI()

@app.get('/')
def home():
    return {'ciao': 'Welcome to the API prediction'}

@app.post('/predict')
def predict(lang: str, request: dict):
    json_response = {}
    product_list = request['product_list']
    input_text = vectorizer(np.array([[product] for product in product_list])).numpy()
    prediction_list, confidence_scores = predict_f(model, input_text, label_classes)
    for product, prediction, confidence_score in zip(product_list, prediction_list, confidence_scores):
        json_response[product] = {"prediction":prediction, "confidence":round(confidence_score.item(), 4)}
    return json_response