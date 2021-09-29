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

def predict(array_product):
    prediction_list = []
    predictions = model.predict(array_product)
    argm_predicts = np.argmax(predictions, axis=1)
    for argm_idx in argm_predicts:
        prediction_list.append(label_classes[argm_idx])
    return prediction_list

def open_target_file(file_path):
    product_list = []
    with open(file_path, 'r') as fp:
        for line in fp:
            product_list.append(line.strip())
    return product_list

def main():
    output_json = {}
    target_file_path = sys.argv[1]
    assert isinstance(target_file_path, str)
    product_list = open_target_file(target_file_path)
    input_text = vectorizer(np.array([[product] for product in product_list])).numpy()
    prediction_list = predict(input_text)
    for product, prediction in zip(product_list, prediction_list):
        output_json[product] = prediction
    print(json.dumps(output_json, indent=2))


global label_classes, model

vocab_data = load_pickle("data/vocab.pickle")
label_classes = load_pickle("data/label_classes.pickle")

vectorizer = TextVectorization(output_sequence_length=MAX_SEQUENCE_LENGTH, vocabulary=vocab_data)
model = keras.models.load_model("data/models/model_saved_1")

if __name__ == '__main__':
    main()
