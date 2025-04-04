from flask import Flask
import torch_geometric.nn as tgnn
import torch.nn as tnn
import pickle


gnn_app = Flask(__name__)

@gnn_app.route('/model_api', methods=["POST"])
def model_api():
    pass


@gnn_app.route('/')
def homepage():
    return "bla"

if __name__ == "__main__":
    debug, host, port = True, "127.0.0.1", "8080"
    gnn_app.run(debug=debug, host=host, port=port)