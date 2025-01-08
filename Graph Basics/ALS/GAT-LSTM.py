# -*- coding: utf-8 -*-
"""
ALS-GAT-LSTM

# Feasibility Work
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

def prepare_subject_data_with_validation(S1, threshold=0.1, num_folds=4, fold_config=[1, 0, 0, 0]):
    """
    Prepare subject data with validation, allowing adaptive fold selection for training/testing.

    Parameters:
        S1: Subject data (e.g., EEG signals).
        threshold: Threshold for PLV-based graph edge creation.
        num_folds: Number of folds to split the data into.
        fold_config: List indicating fold usage ([1, 0, ...]), where 1 = train, 0 = test.
    
    Returns:
        train_loader: DataLoader for training set.
        test_loader: DataLoader for testing set.
    """
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

    # Determine train and test data based on fold_config
    train_data = []
    test_data = []

    for i, flag in enumerate(fold_config):
        if flag == 1:
            train_data.extend(folds[i])
        elif flag == 0:
            test_data.extend(folds[i])

    # Create DataLoader for train and test sets
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    return train_loader, test_loader


def get_model_memory(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.element_size() * p.numel() for p in model.parameters()) + \
        sum(b.element_size() * b.numel() for b in model.buffers())
    return total_params, total_size / (1024 ** 2)  # Return memory in MB


def get_ram_usage():
    process = psutil.Process()  # Current process
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 3)  # Return RAM usage in GB

class GATLSTM(nn.Module):
    def __init__(self, in_channels, gat_out_channels, lstm_hidden_size, num_classes, num_heads, num_lstm_layers):
        super(GATLSTM, self).__init__()

        # GAT module
        self.gat = GATv2Conv(in_channels, gat_out_channels, heads=num_heads, concat=False)

        # LSTM module
        self.lstm = nn.LSTM(input_size=gat_out_channels, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=num_lstm_layers, 
                            batch_first=True, dropout=0.5)

        # Fully connected layer
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, batch):
        # Extract batch components
        x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch

        # Apply GAT to node features
        node_embeddings = self.gat(x, edge_index)

        # Pool node embeddings to get graph-level embeddings
        graph_embeddings = global_mean_pool(node_embeddings, batch_index)

        # Reshape for LSTM input (batch_size, sequence_length=1, gat_out_channels)
        graph_embeddings = graph_embeddings.unsqueeze(1)

        # Process through LSTM
        lstm_output, (h_n, c_n) = self.lstm(graph_embeddings)

        # Use the final hidden state for classification
        logits = self.fc(lstm_output[:, -1, :])
        return logits

# % Preparing Data
data_dir = 'C:/Users/uceerjp/Desktop/PhD/Multi-session Data/OG_Full_Data'
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]
#subject_numbers = [1]
subject_data = {}
all_subjects_accuracies = {}
fold_accuracies = []

# Initialize a dictionary to store accuracies for each subject
accuracy_history = {}

# Loop through subjects
for subject_number in subject_numbers:
    # Initialize a dictionary to store accuracies for this subject
    accuracy_history[subject_number] = {"train": [], "test": []}

    mat_fname = pjoin(data_dir, f'OGFS{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    S1 = mat_contents[f'Subject{subject_number}'][:, :]

    # Prepare train and test loaders for this subject
    train_loader, test_loader = prepare_subject_data_with_validation(S1, threshold=0, num_folds=4, fold_config=[1, 1, 0, 0])

    # Initialize the model, criterion, and optimizer
    model = GATLSTM(in_channels=22, gat_out_channels=32, lstm_hidden_size=64, 
                    num_classes=2, num_heads=8, num_lstm_layers=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    num_epochs = 200

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_correct = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            loss.backward()
            optimizer.step()

            # Update training accuracy
            train_correct += (outputs.argmax(dim=1) == batch.y).sum().item()

        train_accuracy = train_correct / len(train_loader.dataset) * 100
        accuracy_history[subject_number]["train"].append(train_accuracy)  # Append train accuracy

        # Evaluation phase
        model.eval()
        test_correct = 0

        with torch.no_grad():
            for batch in test_loader:
                outputs = model(batch)
                test_correct += (outputs.argmax(dim=1) == batch.y).sum().item()

        test_accuracy = test_correct / len(test_loader.dataset) * 100
        accuracy_history[subject_number]["test"].append(test_accuracy)  # Append test accuracy

        #scheduler.step()
        
        print(f"{epoch}: Train: {train_accuracy:.2f}%, Test: {test_accuracy:.2f}%")
    
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy_history[subject_number]["train"], label='Train Accuracy', color='blue')
    plt.plot(accuracy_history[subject_number]["test"], label='Test Accuracy', color='orange')
    plt.title(f"Accuracy Over Epochs for Subject {subject_number}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot the accuracy for each subject
for subject_number in subject_numbers:
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy_history[subject_number]["train"], label='Train Accuracy', color='blue')
    plt.plot(accuracy_history[subject_number]["test"], label='Test Accuracy', color='orange')
    plt.title(f"Accuracy Over Epochs for Subject {subject_number}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.show()


