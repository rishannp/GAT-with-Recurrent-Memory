# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:26:26 2024
Rishan Patel, UCL, Bioelectronics Group.

KL Div Loss

https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
"""
import pandas as pd
import seaborn as sns
import os
from os.path import dirname, join as pjoin
import scipy as sp
import scipy.io as sio
from scipy import signal
import numpy as np
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import networkx as nx
import torch as torch
from scipy.signal import welch
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif
from scipy.integrate import simpson
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GAT, GraphNorm
from torch_geometric.nn import global_mean_pool
from torch import nn
import pickle
from progressbar import progressbar
import time
from torch_geometric.explain import Explainer, GNNExplainer
import psutil
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# % Functions


def plvfcn(eegData):
    numElectrodes = eegData.shape[1]
    numTimeSteps = eegData.shape[0]
    plvMatrix = np.zeros((numElectrodes, numElectrodes))
    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            phase1 = np.angle(sig.hilbert(eegData[:, electrode1]))
            phase2 = np.angle(sig.hilbert(eegData[:, electrode2]))
            phase_difference = phase2 - phase1
            plv = np.abs(np.sum(np.exp(1j * phase_difference)) / numTimeSteps)
            plvMatrix[electrode1, electrode2] = plv
            plvMatrix[electrode2, electrode1] = plv
    return plvMatrix


def compute_plv(subject_data):
    idx = ['L', 'R']
    numElectrodes = subject_data['L'][0, 1].shape[1]
    plv = {field: np.zeros(
        (numElectrodes, numElectrodes, subject_data.shape[1])) for field in idx}
    for i, field in enumerate(idx):
        for j in range(subject_data.shape[1]):
            x = subject_data[field][0, j]
            plv[field][:, :, j] = plvfcn(x)
    l, r = plv['L'], plv['R']
    yl, yr = np.zeros((subject_data.shape[1], 1)), np.ones(
        (subject_data.shape[1], 1))
    img = np.concatenate((l, r), axis=2)
    y = np.concatenate((yl, yr), axis=0)
    y = torch.tensor(y, dtype=torch.long)
    return img, y

def create_graphs(plv, threshold):

    graphs = []
    for i in range(plv.shape[2]):
        G = nx.Graph()
        G.add_nodes_from(range(plv.shape[0]))
        for u in range(plv.shape[0]):
            for v in range(plv.shape[0]):
                if u != v and plv[u, v, i] > threshold:
                    G.add_edge(u, v, weight=plv[u, v, i])
        graphs.append(G)
    return graphs

def prepare_subject_data_with_validation(S1, threshold=0.1, num_folds=4):
    # Compute PLV and create graphs
    plv, y = compute_plv(S1)
    graphs = create_graphs(plv, threshold)
    
    # Get the number of electrodes and initialize adjacency matrix
    numElectrodes = S1['L'][0, 1].shape[1]
    adj = np.zeros([numElectrodes, numElectrodes, len(graphs)])
    
    for i, G in enumerate(graphs):
        adj[:, :, i] = nx.to_numpy_array(G)

    adj = torch.tensor(adj, dtype=torch.float32)

    # Initialize edge indices list
    edge_indices = []

    # Iterate over the adjacency matrices to create edge indices
    for i in range(adj.shape[2]):
        source_nodes = []
        target_nodes = []

        for row in range(adj.shape[0]):
            for col in range(adj.shape[1]):
                if adj[row, col, i] >= threshold:
                    source_nodes.append(row)
                    target_nodes.append(col)
                else:
                    source_nodes.append(0)
                    target_nodes.append(0)

        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        edge_indices.append(edge_index)

    # Stack edge indices
    edge_indices = torch.stack(edge_indices, dim=-1)

    # Prepare data for each graph
    data_list = []
    for i in range(np.size(adj, 2)):
        data_list.append(Data(x=adj[:, :, i], edge_index=edge_indices[:, :, i], y=y[i, 0]))

    # Split data into training and testing folds
    datal = []
    datar = []
    size = len(data_list)
    idx = size // 2
    c = [0, idx, idx * 2, idx * 3]

    datal = data_list[c[0]:c[1]]
    datar = data_list[c[1]:c[2]]

    data_list = []

    for i in range(idx):
        data_list.extend([datal[i], datar[i]])

    # Split data into N folds
    fold_size = len(data_list) // num_folds
    folds = [data_list[i * fold_size:(i + 1) * fold_size] for i in range(num_folds)]

    session_1 = folds[0]
    session_2 = folds[1]
    
    # Split session 2 into validation and testing sets (20% for validation, 80% for testing)
    val_split = int(0.2 * len(session_2))  # 20% for validation

    val_set = session_2[:val_split]
    test_set = session_2[val_split:]

    # Create DataLoader for train, validation, and test sets
    train_loader = DataLoader(session_1, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

    return train_loader, val_loader, test_loader

def get_model_memory(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.element_size() * p.numel() for p in model.parameters()) + \
        sum(b.element_size() * b.numel() for b in model.buffers())
    return total_params, total_size / (1024 ** 2)  # Return memory in MB


def get_ram_usage():
    process = psutil.Process()  # Current process
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 3)  # Return RAM usage in GB

class GAT(nn.Module):
    def __init__(self, hidden_channels, heads):
        super(GAT, self).__init__()

        # Define GAT convolution layers
        self.conv1 = GATv2Conv(
            22, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATv2Conv(
            hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        self.conv3 = GATv2Conv(
            hidden_channels * heads, hidden_channels, heads=heads, concat=True)

        # Define GraphNorm layers
        self.gn1 = GraphNorm(hidden_channels * heads)
        self.gn2 = GraphNorm(hidden_channels * heads)
        self.gn3 = GraphNorm(hidden_channels * heads)

        # Define the final linear layer
        self.lin = nn.Linear(hidden_channels * heads, 2)

    def forward(self, x, edge_index, batch, return_attention_weights=True):
        # print(f"Batch size: {batch.size()}")  # Check the batch tensor

        # Apply first GAT layer and normalization
        w1, att1 = self.conv1(
            x, edge_index, return_attention_weights=True)
        w1 = F.relu(w1)
        w1 = self.gn1(w1, batch)  # Apply GraphNorm

        # Apply second GAT layer and normalization
        w2, att2 = self.conv2(
            w1, edge_index, return_attention_weights=True)
        w2 = F.relu(w2)
        w2 = self.gn2(w2, batch)  # Apply GraphNorm

        # Apply third GAT layer and normalization
        w3, att3 = self.conv3(
            w2, edge_index, return_attention_weights=True)
        w3 = self.gn3(w3, batch)  # Apply GraphNorm

        # print(f"Shape of w3 before pooling: {w3.size()}")  # Before global mean pooling

        # Global pooling
        w3 = global_mean_pool(w3, batch)
        # print(f"Shape of w3 after pooling: {w3.size()}")  # After global mean pooling

        # Apply dropout and final classifier
        w3 = F.dropout(w3, p=0.50, training=self.training)
        o = self.lin(w3)

        return o, w3, att3
    
# Define the training function
def train():
    model.train()
    start_time = time.time()  # Start timing
    all_edge = []
    all_att = []  # List to accumulate attention scores for all batches
    all_w = []
    correct = 0  # Counter for correct predictions
    total = 0  # Counter for total predictions

    for data in train_loader:
        # Get model output
        out, w3, att3 = model(data.x, data.edge_index, data.batch)

        # Append the attention scores for the current batch to the list
        all_att.append(att3[1])
        all_edge.append(att3[0])
        all_w.append(w3)

        # Convert output to log probabilities for KL Divergence loss
        log_out = F.log_softmax(out, dim=1)
        
        # Convert target to one-hot encoding (assuming data.y are class indices)
        target = F.one_hot(data.y, num_classes=out.size(1)).float()

        # Calculate KLDivLoss
        loss = F.kl_div(log_out, target, reduction='batchmean')  # KL Divergence Loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Calculate accuracy for the current batch
        pred = out.argmax(dim=1)  # Get the predicted class (index with highest probability)
        correct += (pred == data.y).sum().item()  # Count correct predictions
        total += data.y.size(0)  # Increment total by the number of samples in the batch

    end_time = time.time()  # End timing
    # Concatenate attention scores from all batches
    all_att = torch.cat(all_att, dim=0)
    all_edge = torch.cat(all_edge, dim=1)
    all_w = torch.cat(all_w, dim=0)

    # Calculate overall accuracy for the epoch
    train_accuracy = correct / total * 100  # Convert to percentage
    return end_time - start_time, train_accuracy, all_w, all_att, all_edge


def test(loader):
    model.eval()
    correct = 0
    inference_start = time.time()  # Start timing
    all_edge = []
    all_w = []
    all_att = []  # List to accumulate attention scores for all batches
    for data in loader:
        out, w3, att3 = model(data.x, data.edge_index, data.batch)

        # Append the attention scores for the current batch to the list
        all_att.append(att3[1])
        all_edge.append(att3[0])
        all_w.append(w3)

        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    inference_end = time.time()  # End timing
    inference_time = inference_end - inference_start  # Calculate inference time

    # Concatenate attention scores from all batches
    all_att = torch.cat(all_att, dim=0)
    all_edge = torch.cat(all_edge, dim=1)
    all_w = torch.cat(all_w, dim=0)

    return correct / len(loader.dataset)*100, inference_time, all_w, all_att, all_edge


# % Preparing Data
data_dir = 'C:/Users/uceerjp/Desktop/PhD/Multi-session Data/OG_Full_Data'
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]
subject_data = {}
all_subjects_accuracies = {}

# Initialize fold_accuracies outside the subject loop
fold_accuracies = []

# Loop through subjects
for subject_number in subject_numbers:
    # Initialize lists to store accuracies for each epoch
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    
    mat_fname = pjoin(data_dir, f'OGFS{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    S1 = mat_contents[f'Subject{subject_number}'][:, :]

    # Prepare train and test loaders for this subject
    train_loader, val_loader, test_loader = prepare_subject_data_with_validation(S1)

    # Initialize the model, optimizer, and loss function
    model = GAT(hidden_channels=22, heads=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()  # We'll not use criterion here, KLDivLoss will be used instead

    optimal_fold_acc = 0  # Store only the accuracy (float)
    fold_test_accuracies = []
    fold_weights = {}
    fold_att = {}
    train_times = []
    inference_times = []
    ram = []

    start_train_time = time.time()  # Start timing for the entire training process

    for epoch in range(1, 500):  # Note: Range is 1 to 500 (inclusive)
        epoch_train_time, train_acc, trainw3, trainatt3, trainedge3 = train()
        train_accuracies.append(train_acc)  # Store train accuracy

        val_acc, inference_time, valw3, valatt3, valedge3 = test(val_loader)
        val_accuracies.append(val_acc)  # Store validation accuracy

        test_acc, inference_time, testw3, testatt3, testedge3 = test(test_loader)
        test_accuracies.append(test_acc)  # Store test accuracy

        inference_times.append(inference_time)

        ram_usage = get_ram_usage()

        fold_test_accuracies.append(test_acc)

        print(f"Epoch {epoch}: Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f},  Test Accuracy: {test_acc:.4f}")

        if test_acc > optimal_fold_acc:
            optimal_fold_acc = test_acc

            best_weights = {
                'w3': trainw3.detach().cpu().numpy(),
            }
            best_att = {
                'att3': testatt3.detach().cpu().numpy(),
                'edge3': testedge3.detach().cpu().numpy(),
            }

            torch.save(model.state_dict(), f"best_model_{subject_number}.pth")

    end_train_time = time.time()
    total_train_time = end_train_time - start_train_time

    fold_accuracies.append({
        'subject': subject_number,
        'optimal': optimal_fold_acc,
        'mean': np.mean(fold_test_accuracies),
        'high': np.max(fold_test_accuracies),
        'low': np.min(fold_test_accuracies),
        'total_train_time': total_train_time,
        'avg_train_time': np.mean(train_times),
        'avg_inference_time': np.mean(inference_times),
        'best_weights': best_weights,
        'best_attention_scores': best_att,
        'ram': ram_usage
    })

    all_subjects_accuracies[f'S{subject_number}'] = fold_accuracies
    
    # Plotting the accuracies over epochs
    plt.figure(figsize=(10, 6))

    # Plot each of the accuracies
    plt.plot(train_accuracies, label='Train Accuracy', color='blue', linewidth=2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green', linewidth=2)
    plt.plot(test_accuracies, label='Test Accuracy', color='red', linewidth=2)

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Epochs for Subject {subject_number}')
    plt.legend()

    # Show the plot
    plt.show()

