# AI Models for categorization

This repo contains a script to classify raw data into the appropriate categories used for the subsequent calculations.

## Installation
```bash
pip install -r requirements.txt
```

## Usage

Example of usage:
```bash
python predict.py products.txt -l en
```
This example assumes that:
1. you have a file 'products.txt' with raw data you want to make predictions about.
2. the examples are in english.

Just change the positional argument to whatever the path to your file is. Then use the '-l' flag to indicate the language.

**N.B.** If you're not sure about which values you can use for the '-l' flag, run `python predict.py -h` to have a list of possible choices.
