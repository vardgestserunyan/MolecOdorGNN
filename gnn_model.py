from flask import Flask, render_template, request
import torch_geometric.utils as tguitls
import torch_geometric.loader as tgloader
from datetime import datetime
from odor_from_smiles import GraphNN
import pickle

app = Flask(__name__)
with open('apple_odor_gnn.pkl', "rb") as file:
    gnn_model = pickle.load(file)


@app.route('/predictor', methods=["GET", "POST"])
def predictor():
    data, show_results = {}, False
    if request.method == "POST":
        smiles_input = request.form.get('smiles_input')
        smiles_input_list = smiles_input.splitlines()
        graph_obj_list = []
        for line in smiles_input_list:
            graph_obj = tguitls.from_smiles(line)
            graph_obj_list.append(graph_obj)
        graph_input_batch = tgloader.DataLoader(graph_obj_list, batch_size=len(graph_obj_list))
        pred_logits = gnn_model( next(iter(graph_input_batch)) )
        prediction = pred_logits.argmax(dim=1).tolist()
        date, time = datetime.now().date(), datetime.now().time()
        data = {'user_input': smiles_input,
                'smiles_input_list':smiles_input_list,
                'prediction':prediction,
                'date':date,
                'time':time
                }
        show_results = True
    return render_template('/predictor.html', data=data, show_results=show_results)


if __name__ == "__main__":
    host, port, debug = "127.0.0.1", "8080", True
    app.run(host=host, port=port, debug=debug)