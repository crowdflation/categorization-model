import json
import numpy as np
import pickle
import sys
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization


MAX_SEQUENCE_LENGTH = 50

global label_classes, model

with open("data/vocab.pickle", "rb") as fp:
    vocab_data = pickle.load(fp)
with open("data/label_classes.pickle", "rb") as fp:
    label_classes = pickle.load(fp)

vectorizer = TextVectorization(output_sequence_length=MAX_SEQUENCE_LENGTH, vocabulary=vocab_data)
model = keras.models.load_model("models/model_saved_1")

def predict(input_text):
    prediction = model.predict(input_text)
    argm_predict = np.argmax(prediction, axis=1)
    category = label_classes[argm_predict[0]]
    return category

if __name__ == '__main__':
    text = sys.argv[1]
    assert isinstance(text, str)
    input_text = vectorizer(np.array([[text]])).numpy()
    category = predict(input_text)
    prediction = {'product_name': text, 'category_prediction': category}
    print(json.dumps(prediction, indent=2))
