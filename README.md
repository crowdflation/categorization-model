# AI Models for categorization

This repo contains a script to classify raw data into the appropriate categories used for the subsequent calculations.

## Installation
```bash
pip install -r requirements.txt
```

## Usage

Example of usage:
```bash
python predict.py products.txt -c food_b
```
This example assumes that:
1. you have a file 'products.txt' with raw data you want to make predictions about.
2. the classes that you want to categorize the data into, are the ones belonging to the food&bevarage area.

Just change the positional argument to whatever the path to your file is, and use the '-c' flag to indicate which category that data belong to.

**N.B.** If you're not sure about what values you can use for the '-c' flag, run `python predict.py -h` to have a list of possible choices.
