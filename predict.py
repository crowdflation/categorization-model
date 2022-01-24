import argparse
import json
import numpy as np
import pickle
import yaml
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization


MAX_SEQUENCE_LENGTH = 50

def load_config():
    with open("data/config.yaml", "r") as yaml_config:
        return yaml.safe_load(yaml_config)

def load_pickle(file_path: str):
    with open(file_path, "rb") as fp:
        return pickle.load(fp)

def predict(array_product: np.ndarray):
    prediction_list = []
    predictions = model.predict(array_product)
    argm_predicts = np.argmax(predictions, axis=1)
    for argm_idx in argm_predicts:
        prediction_list.append(label_classes[argm_idx])
    return prediction_list

def open_target_file(file_path: str):
    product_list = []
    with open(file_path, 'r', encoding = 'cp850') as fp:
        for line in fp:
            product_list.append(line.strip())
    return product_list

def init_global_obj(category, lang):
    config = load_config()
    global label_classes, model, vectorizer
    vocab_data = load_pickle(config['vocab'][lang][category])
    label_classes = load_pickle(config['label_classes'][lang][category])
    model = keras.models.load_model(config['model'][lang][category])
    vectorizer = TextVectorization(output_sequence_length=MAX_SEQUENCE_LENGTH,
                                   vocabulary=vocab_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, metavar='path',
                        help="path to the target file that we want to get predictions from")
    parser.add_argument("-c", "--category", required=True, type=str,
                        help="the major category which the input data belong to",
                        choices=['food_bev','apparel'])
    parser.add_argument("-l", "--language", required=True, type=str,
                        help="the language of the input data",
                        choices=['en','tr'])
    args = parser.parse_args()

    category = args.category
    language = args.language
    init_global_obj(category, language)

    output_json = {}
    target_file_path = args.data
    product_list = open_target_file(target_file_path)
    input_text = vectorizer(np.array([[product] for product in product_list])).numpy()
    prediction_list = predict(input_text)
    for product, prediction in zip(product_list, prediction_list):
        output_json[product] = prediction
    print(json.dumps(output_json, indent=2))


if __name__ == '__main__':
    main()
