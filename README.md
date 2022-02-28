# AI Models for categorization

This repo contains the code needed to run a server that expose APIs for predictions. We use predictions to classify raw data into the appropriate categories used for the subsequent calculations.

## Installation
```bash
pip install -r requirements.txt
```

## Usage

Example of usage:
```bash
uvicorn main:api
```
This command will start the server. 
There will be a line telling where the app is being served. Default should be on your local machine on port 8000.

You can use the "--host" and "--port" flags to set these for your server. For example:
```bash
uvicorn main:api --host 0.0.0.0 --port 80
```

If you want to take a look at the API docs, you can conveniently head to "{wherever_you_are_serving_the_app}/docs" to explore and even play around with the APIs.
So, in case of your local machine, that would be "http://127.0.0.1:8000/docs"
