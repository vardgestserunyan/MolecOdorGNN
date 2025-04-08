import torch_geometric.utils as tguitls
import torch_geometric.loader as tgloader
from odor_from_smiles import GraphNN
import pickle

with open('apple_odor_gnn.pkl', "rb") as file:
    gnn_model = pickle.load(file)

def gnn_runner(smiles_input_list):
    graph_obj_list = []
    for line in smiles_input_list:
        graph_obj = tguitls.from_smiles(line)
        graph_obj_list.append(graph_obj)
    graph_input_batch = tgloader.DataLoader(graph_obj_list, batch_size=len(graph_obj_list))
    pred_logits = gnn_model( next(iter(graph_input_batch)) )
    prediction = pred_logits.argmax(dim=1).tolist()

    return prediction