import json
import numpy as np
import pickle
import sys
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization


MAX_SEQUENCE_LENGTH = 50

def load_pickle(file_path):
    with open(file_path, "rb") as fp:
        loaded_file = pickle.load(fp)
    return loaded_file

def predict(input_text):
    prediction = model.predict(input_text)
    argm_predict = np.argmax(prediction, axis=1)
    category = label_classes[argm_predict[0]]
    return category

global label_classes, model

vocab_data = load_pickle("data/vocab.pickle")
label_classes = load_pickle("data/label_classes.pickle")

vectorizer = TextVectorization(output_sequence_length=MAX_SEQUENCE_LENGTH, vocabulary=vocab_data)
model = keras.models.load_model("data/models/model_saved_1")

if __name__ == '__main__':
    text = sys.argv[1]
    assert isinstance(text, str)
    input_text = vectorizer(np.array([[text]])).numpy()
    category = predict(input_text)
    prediction = {text: category}
    print(json.dumps(prediction, indent=2))
