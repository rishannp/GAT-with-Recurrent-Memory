# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:26:26 2024
Rishan Patel, UCL, Bioelectronics Group.

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
import higher

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

    # Assign folds to sessions
    session_1 = folds[0]  # First fold is used for training
    session_2 = [data for fold in folds[1:] for data in fold]  # Flatten remaining folds

    # # Split session 2 into validation and testing sets (20% for validation, 80% for testing)
    # val_split = int(0.2 * len(session_2))  # 20% for validation
    # val_set = session_2[:val_split]
    # test_set = session_2[val_split:]

    # Create DataLoader for train, validation, and test sets
    train_loader = DataLoader(session_1, batch_size=32, shuffle=False)
    # val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    test_loader = DataLoader(session_2, batch_size=32, shuffle=False)

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


class GATMeta(nn.Module):
    def __init__(self, hidden_channels, heads):
        super(GATMeta, self).__init__()

        # Define GAT layers as before
        self.conv1 = GATv2Conv(
            22, hidden_channels, heads=heads, concat=False)
        self.conv2 = GATv2Conv(
            hidden_channels * heads, hidden_channels, heads=heads, concat=False)
        self.conv3 = GATv2Conv(
            hidden_channels * heads, hidden_channels, heads=heads, concat=False)

        self.gn1 = GraphNorm(hidden_channels * heads)
        self.gn2 = GraphNorm(hidden_channels * heads)
        self.gn3 = GraphNorm(hidden_channels * heads)

        self.lin = nn.Linear(hidden_channels * heads, 2)

    def forward(self, x, edge_index, batch):
        w1, _ = self.conv1(x, edge_index, return_attention_weights=True)
        w1 = F.relu(w1)
        w1 = self.gn1(w1, batch)

        w2, _ = self.conv2(w1, edge_index, return_attention_weights=True)
        w2 = F.relu(w2)
        w2 = self.gn2(w2, batch)

        w3, _ = self.conv3(w2, edge_index, return_attention_weights=True)
        w3 = self.gn3(w3, batch)

        w3 = global_mean_pool(w3, batch)

        w3 = F.dropout(w3, p=0.50, training=self.training)
        o = self.lin(w3)
        return o, w3


def create_model(hidden_channels=22, heads=1):
    """
    Create and return a fresh instance of the GATMeta model.
    
    hidden_channels: The number of hidden channels for GAT layers (default 22).
    heads: The number of attention heads for GAT layers (default 1).
    
    Returns:
    A new instance of the GATMeta model.
    """
    return GATMeta(hidden_channels=hidden_channels, heads=heads)



def calculate_accuracy(predictions, labels):
    _, predicted_classes = predictions.max(dim=1)
    correct = (predicted_classes == labels).sum().item()
    accuracy = correct / len(labels)
    return accuracy

def meta_train_loop(train_loader, test_loader, model, optimizer, num_epochs=100, inner_lr=0.01, outer_lr=0.001, num_inner_steps=1):
    """
    Meta-train the model using MAML and track loss/accuracy for training, validation, and test sets.
    
    train_loader: DataLoader for training data
    val_loader: DataLoader for validation data
    test_loader: DataLoader for testing data
    model: the GAT model
    optimizer: optimizer for the outer loop
    num_epochs: number of epochs to train the meta-model
    inner_lr: learning rate for the inner loop (task-specific adaptation)
    outer_lr: learning rate for the outer loop (meta-update)
    num_inner_steps: number of gradient steps for task-specific adaptation
    """
    # Lists to track loss and accuracy for each set
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Outer loop (meta-update)
    for epoch in range(num_epochs):
        total_params, model_memory = get_model_memory(model)
        ram_usage = get_ram_usage()
        
        model.train()
        meta_train_loss = 0
        meta_train_accuracy = 0
        
        # Loop over tasks in the train_loader (assuming you're using a batch of tasks)
        for data in train_loader:
            # Initialize a list to hold gradients for meta-update
            meta_gradients = []
            task_losses = []
            task_accuracies = []
            
            # Task-specific adaptation (inner loop)
            for task in range(len(data)):
                task_data = data[task]
                task_model = create_model()  # Create a fresh model for each task
                
                # Adapt the model to the task using gradient steps (inner loop)
                task_optimizer = torch.optim.SGD(task_model.parameters(), lr=inner_lr)
                task_optimizer.zero_grad()
                
                # Forward pass and loss computation for the task
                task_output, _ = task_model(task_data.x, task_data.edge_index, task_data.batch)
                task_loss = F.cross_entropy(task_output, task_data.y)
                task_loss.backward()  # Compute gradients
                
                # Calculate accuracy for the task
                task_accuracy = calculate_accuracy(task_output, task_data.y)
                
                # Store the task-specific gradients and losses
                meta_gradients.append([p.grad for p in task_model.parameters()])
                task_losses.append(task_loss.item())
                task_accuracies.append(task_accuracy)
                
                # Update the model for the task (inner loop)
                task_optimizer.step()

            # Meta-update (outer loop)
            optimizer.zero_grad()
            for param, task_grad in zip(model.parameters(), zip(*meta_gradients)):
                # Average gradients across tasks
                avg_grad = torch.mean(torch.stack(task_grad), dim=0)
                param.grad = avg_grad  # Assign averaged gradients to the model
            
            # Update the meta-model with the averaged gradients
            optimizer.step()

            # Calculate the meta-training loss and accuracy
            meta_train_loss += sum(task_losses) / len(task_losses)
            meta_train_accuracy += sum(task_accuracies) / len(task_accuracies)
        
        # Append train loss and accuracy for this epoch
        train_losses.append(meta_train_loss / len(train_loader))
        train_accuracies.append(meta_train_accuracy / len(train_loader))
        
        # Validation and Testing
        test_loss, test_accuracy = test_model(model, test_loader)
        
        # Append val and test losses and accuracies
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Accuracy: {meta_train_accuracy / len(train_loader) * 100:.2f}%, "
              f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Plot Loss and Accuracy
    plot_loss_accuracy(train_losses, test_losses, train_accuracies, test_accuracies)
    print(f"Total Params: {total_params}, Model Memory: {model_memory:.2f} MB, RAM Usage: {ram_usage:.2f} GB")

# Function for validation
def validate_model(model, val_loader):
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for data in val_loader:
            out, _ = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            pred = out.argmax(dim=1)
            val_accuracy += (pred == data.y).sum().item()
            val_loss += loss.item()
    val_loss /= len(val_loader.dataset)
    val_accuracy /= len(val_loader.dataset)
    return val_loss, val_accuracy

# Function for testing
def test_model(model, test_loader):
    model.eval()
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            out, _ = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            pred = out.argmax(dim=1)
            test_accuracy += (pred == data.y).sum().item()
            test_loss += loss.item()
    test_loss /= len(test_loader.dataset)
    test_accuracy /= len(test_loader.dataset)
    return test_loss, test_accuracy


# Plotting Loss and Accuracy
def plot_loss_accuracy(train_losses, test_losses, train_accuracies, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Loop for each subject's data
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]
for subject_number in subject_numbers:
    # Load and prepare data
    data_dir = 'C:/Users/uceerjp/Desktop/PhD/Multi-session Data/OG_Full_Data'
    mat_fname = pjoin(data_dir, f'OGFS{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    S1 = mat_contents[f'Subject{subject_number}'][:, :]

    # Prepare DataLoader (training, validation, testing)
    train_loader, test_loader = prepare_subject_data_with_validation(S1, threshold=0.01, num_folds=4)
    
    # Initialize model, optimizer
    model = GATMeta(hidden_channels=22, heads=1)  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "max", factor = 0.1, patience=10)
    
    # Meta-train the model for this subject
    meta_train_loop(train_loader, test_loader, model, optimizer, num_epochs=200)