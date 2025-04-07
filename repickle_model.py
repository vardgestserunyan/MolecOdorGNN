from odor_from_smiles import GraphNN
import pickle
import os

with open("apple_odor_gnn.pkl", "rb") as file:
    gnn_model = pickle.load(file)

os.remove("apple_odor_gnn.pkl")
with open("apple_odor_gnn.pkl", "wb") as file:
    pickle.dump(gnn_model, file)