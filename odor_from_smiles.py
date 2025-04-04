import kagglehub
import time
import os
import pandas as pd
import numpy as np
import torch_geometric.utils as tgutils
import torch_geometric.loader as tgloader
import torch_geometric.nn as tgnn
import torch_geometric.transforms as tgtrans
import torch.optim as toptim
import pickle
import torch.nn as nn
from torch import manual_seed, tensor, float32, int64, long
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight, class_weight
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

manual_seed(12121995)
## Loading the dataset
dirpath = kagglehub.dataset_download("aryanamitbarsainyan/multi-labelled-smiles-odors-dataset")
filename = "Multi-Labelled_Smiles_Odors_dataset.csv"
fullpath = dirpath+"/"+filename
while not os.path.exists(fullpath):
    time.sleep(.5)
raw_data_df = pd.read_csv(fullpath)

# Prepping the dataset, splitting, computing class weights for balanced training
raw_data_df.drop("descriptors", axis=1, inplace=True)
raw_data_df["graph_struct"] = raw_data_df.apply(lambda row: tgutils.from_smiles(row["nonStereoSMILES"]), axis=1)

def modify_graph(row):
    transform = tgtrans.Compose([tgtrans.VirtualNode()])
    graph = transform(row["graph_struct"])
    graph.y_apple = row["apple"]
    return graph 
raw_data_df["graph_struct"] = raw_data_df.apply(modify_graph, axis=1)

# Define the train/test split and class weights
x_train, x_test, y_train, y_test = train_test_split(raw_data_df["graph_struct"], raw_data_df["apple"], 
                                                    test_size=0.2, stratify=raw_data_df["apple"], random_state=12121995)
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)

# Define the model
class GraphNN(nn.Module):
    def __init__(self, graph_breadth, graph_depth, lin_breadth, lin_depth, device):
        super().__init__()
        graph_layers_list = [(tgnn.GCNConv(-1,graph_breadth), 'x, edge_index -> x'),
                             (nn.ReLU(inplace=True), 'x -> x')]
        for _ in range(graph_depth):
            graph_layers_list.append( (tgnn.GCNConv(graph_breadth,graph_breadth), 'x, edge_index -> x') )
            graph_layers_list.append( (tgnn.norm.BatchNorm(graph_breadth), 'x -> x') )
            graph_layers_list.append( (nn.ReLU(inplace=True), 'x -> x') )

        linear_layers_list = [nn.Linear(graph_breadth,lin_breadth),
                              nn.ReLU(inplace=True)]
        for _ in range(lin_depth):
            linear_layers_list.append(nn.Linear(lin_breadth,lin_breadth))
            linear_layers_list.append(nn.BatchNorm1d(lin_breadth))
            linear_layers_list.append(nn.ReLU(inplace=True))
        linear_layers_list.append(nn.Linear(lin_breadth,2))


        self.graph_layers = tgnn.Sequential('x, edge_index', graph_layers_list)
        self.pooling = tgnn.pool.global_add_pool
        self.linear = nn.Sequential(*linear_layers_list)
        self.device = device
        self.to(device)
        
    def forward(self, minibatch):
        x, edge_index = (minibatch.x).to(float32), (minibatch.edge_index).to(int64)
        h_graph = self.graph_layers(x, edge_index)
        h_pool = self.pooling(h_graph, minibatch.batch)
        output = self.linear(h_pool)
        return output
        
    def trainer(self, input_batch, loss_fcn, optimizer):
        self.train()
        for minibatch in input_batch:
            optimizer.zero_grad()
            y = (minibatch.y_apple).to(long)
            pred = self(minibatch)
            loss = loss_fcn(pred, y)
            loss.backward()
            optimizer.step()

    def tester(self, input_batch, loss_fcn):
        self.eval()
        sum_loss, sum_f1 = 0, 0
        for minibatch in input_batch:
            optimizer.zero_grad()
            y = (minibatch.y_apple).to(int64)
            pred = self(minibatch)
            loss = loss_fcn(pred, y)
            sum_loss += loss
            sum_f1 += f1_score(y.detach().numpy(), (pred.argmax(axis=1).detach()).numpy(), average='binary')
        avg_loss = sum_loss/len(input_batch)
        avg_f1 = sum_f1/len(input_batch)
        
        return avg_loss, avg_f1





batch_size, num_epochs, lr, weight_decay = 200, 100, 5e-4, 1e-4
graph_breadth, graph_depth, lin_breadth, lin_depth, device = 64, 10, 32, 3, "cpu" 
x_train_loader = tgloader.DataLoader( list(x_train), batch_size=batch_size, shuffle=True)
x_test_loader = tgloader.DataLoader( list(x_test), batch_size=batch_size, shuffle=True)
GraphNN_obj = GraphNN(graph_breadth, graph_depth, lin_breadth, lin_depth, device)
loss_fcn = nn.CrossEntropyLoss(weight=tensor(class_weights, dtype=float32))
optimizer = toptim.Adam(GraphNN_obj.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = toptim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

train_loss_agg, test_loss_agg = -1*np.ones(num_epochs), -1*np.ones(num_epochs)
train_f1_agg, test_f1_agg = -1*np.ones(num_epochs), -1*np.ones(num_epochs)
for epoch in tqdm(range(num_epochs), desc="Training Epochs", ncols=100):
     GraphNN_obj.trainer(x_train_loader, loss_fcn, optimizer)
     train_loss, train_f1 = GraphNN_obj.tester(x_train_loader, loss_fcn)
     test_loss, test_f1 = GraphNN_obj.tester(x_test_loader, loss_fcn)
     train_loss_agg[epoch], test_loss_agg[epoch] = train_loss, test_loss
     train_f1_agg[epoch], test_f1_agg[epoch] = train_f1, test_f1
     scheduler.step()



with open("apple_odor_gnn.pkl", "wb") as file:
    pickle.dump(GraphNN_obj, file)
     
     

fig, ax = plt.subplots(figsize=(12,6), nrows=1, ncols=2)
ax[0].plot(range(num_epochs), train_loss_agg)
ax[0].plot(range(num_epochs), test_loss_agg)
ax[0].legend(["Training Set", "Test Set"])

ax[1].plot(range(num_epochs), train_f1_agg)
ax[1].plot(range(num_epochs), test_f1_agg)
ax[1].legend(["Training Set", "Test Set"])

fig.savefig("plot.pdf")
