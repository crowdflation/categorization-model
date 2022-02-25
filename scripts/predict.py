import argparse
import json
import numpy as np
import pickle
import yaml
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization


MAX_SEQUENCE_LENGTH = 50
CONFIG_DEFAULT_PATH = "data/config.yaml"

def load_config(path: str = CONFIG_DEFAULT_PATH):
    with open(path, "r") as yaml_config:
        return yaml.safe_load(yaml_config)

def load_pickle(file_path: str):
    with open(file_path, "rb") as fp:
        return pickle.load(fp)

def load_json(file_path: str):
    with open(file_path, "r") as fp:
        return json.load(fp)

def predict_f(model, array_product: np.ndarray, label_classes:list):
    prediction_list = []
    predictions = model.predict(array_product)
    argm_predicts = np.argmax(predictions, axis=1)
    confidence_scores = np.amax(predictions, axis=1)
    for argm_idx in argm_predicts:
        prediction_list.append(label_classes[argm_idx])
    return prediction_list, confidence_scores

def open_target_file(file_path: str):
    product_list = []
    with open(file_path, 'r', encoding = 'utf8') as fp:
        for line in fp:
            product_list.append(line.strip())
    return product_list

def init_global_obj(lang):
    config = load_config()
    #global label_classes, model, vectorizer
    vocab_data = load_pickle(config['vocab'][lang])
    label_classes = load_json(config['label_classes'][lang])
    model = keras.models.load_model(config['model'][lang])
    vectorizer = TextVectorization(output_sequence_length=MAX_SEQUENCE_LENGTH,
                                   vocabulary=vocab_data)
    return label_classes, model, vectorizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, metavar='path',
                        help="path to the target file that we want to get predictions from")
    parser.add_argument("-l", "--language", required=True, type=str,
                        help="the language of the input data",
                        choices=['en','tr'])
    args = parser.parse_args()

    language = args.language
    label_classes, model, vectorizer = init_global_obj(language)

    output_json = {}
    target_file_path = args.data
    product_list = open_target_file(target_file_path)
    input_text = vectorizer(np.array([[product] for product in product_list])).numpy()
    prediction_list, confidence_scores = predict_f(model, input_text, label_classes)
    for product, prediction, confidence_score in zip(product_list, prediction_list, confidence_scores):
        output_json[product] = {"prediction":prediction, "confidence":round(confidence_score.item(), 4)}
    print(json.dumps(output_json, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
