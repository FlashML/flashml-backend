from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from app import app

from app.model_builder import *

CORS(app)

COUNTER_FILENAME = 'count.txt'
def increment_counter():
    try:
        with open(COUNTER_FILENAME, 'r' ) as f:
            counter = int(f.readline()) + 1
    except FileNotFoundError:
        counter = 1

    with open(COUNTER_FILENAME, 'w' ) as f:
        f.write(str(counter))

@app.route("/api/ping", methods=["GET"])
def ping():
    return "Success"

@app.route("/api/create_code", methods=["GET", "POST"])
def model_builder():
    increment_counter()

    json_data = request.json
    print(json_data)
    # extract the required parameters
    layers = json_data["layers"]
    hyperparameters = json_data["hyperparameters"]
    epochs = hyperparameters["epochs"]
    learning_rate = hyperparameters["learning_rate"]
    momentum = hyperparameters["momentum"]
    batch_size = hyperparameters["batch_size"]
    num_workers = hyperparameters["num_workers"]
    loss = hyperparameters["loss"]
    if loss == "L2":
        loss = "MSELoss"
    elif loss == "L1":
        loss = "L1Loss"
    elif loss == "CE":
        loss = "CrossEntropyLoss"
    dataset_name = json_data["dataset_name"]
    if dataset_name == "FASHIONMNIST":
        dataset_name = "FashionMNIST"
    checkpoint_path = json_data["checkpoint_path"]

    # build the text files
    build_model(layers, batch_size)
    build_dataset(dataset_name, batch_size, num_workers)
    build_training_loop(epochs, learning_rate, momentum, loss, checkpoint_path, dataset_name)

    # create a zip like object
    with ZipFile('data/flashml.zip', 'w') as zipObj:
        # Iterate over all the files in a directory

        zipObj.write('data/train.py', basename('flashML/train.py'))
        zipObj.write('data/model.py', basename('flashML/model.py'))
        zipObj.write('data/README.md', basename('flashML/README.md'))
        zipObj.write('data/requirements.txt', basename('flashML/requirements.txt'))
    print("Success")
    return send_file("../data/flashml.zip")


@app.route("/api/request_train", methods=["GET", "POST"])
def request_train():
    print("Success")
    return send_file("../data/train.py")
