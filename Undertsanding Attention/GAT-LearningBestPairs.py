# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 15:45:55 2024

@author: Rishan Patel, UCL, Bioelectronics and AspireCREATE Groups

TLDR: This code produces: 
    1. Graph representations of Motor Imagery-EEG Signals
    2. Using a attention layer(s), we identify how a GAT finds best electrode pairs
    3. Using Attention Entropy, we find pairs with highest weightings and rank them
    
"""


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
from torch_geometric.data import Data
from torch import nn
import pickle
from progressbar import progressbar
import time
from torch_geometric.explain import Explainer, GNNExplainer
import seaborn as sns

# Function to compute Phase Locking Value (PLV)
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

# Function to compute PLV for the subject data
def compute_plv(subject_data):
    idx = ['L', 'R']
    numElectrodes = subject_data['L'][0, 1].shape[1]
    plv = {field: np.zeros((numElectrodes, numElectrodes, subject_data.shape[1])) for field in idx}
    for i, field in enumerate(idx):
        for j in range(subject_data.shape[1]):
            x = subject_data[field][0, j]
            plv[field][:, :, j] = plvfcn(x)
    l, r = plv['L'], plv['R']
    yl, yr = np.zeros((subject_data.shape[1], 1)), np.ones((subject_data.shape[1], 1))
    img = np.concatenate((l, r), axis=2)
    y = np.concatenate((yl, yr), axis=0)
    y = torch.tensor(y, dtype=torch.long)
    return img, y

# Function to create graphs from PLV matrices
def create_graphs(plv, threshold=0.1):
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

# Function to prepare adjacency matrices and edge indices from graphs
def prepare_graph_data(graphs, threshold=0):
    numElectrodes = graphs[0].number_of_nodes()
    adj = np.zeros([numElectrodes, numElectrodes, len(graphs)])
    edge_indices = []

    for i, G in enumerate(graphs):
        adj[:, :, i] = nx.to_numpy_array(G)
        source_nodes, target_nodes = [], []
        
        for row in range(adj.shape[0]):
            for col in range(adj.shape[1]):
                if adj[row, col, i] >= threshold:
                    source_nodes.append(row)
                    target_nodes.append(col)
        
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        edge_indices.append(edge_index)
    
    edge_indices = torch.stack(edge_indices, dim=-1)
    return adj, edge_indices

# Function to load data for a subject
def load_subject_data(data_dir, subject_number):
    mat_fname = pjoin(data_dir, f'OGFS{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    return mat_contents[f'Subject{subject_number}']

# Function to create Data objects for PyTorch Geometric
def create_data_objects(adj, edge_indices, y):
    data_list = []
    for i in range(adj.shape[2]):
        # Convert adj to a tensor and ensure it's of the right shape for the GAT layer
        x_tensor = torch.tensor(adj[:, :, i], dtype=torch.float)
        edge_index_tensor = edge_indices[:,:,i]
        y_tensor = y[i, 0]
        
        # Create Data object for PyTorch Geometric
        data_list.append(Data(x=x_tensor, edge_index=edge_index_tensor, y=y_tensor))
    return data_list


def create_dataloaders(subject_data_objects, batch_size, shuffle=False):
    """
    Splits subject data into training and testing DataLoaders.

    Parameters:
    - subject_data_objects: list of subject data objects to be split.
    - batch_size: size of the batch for DataLoader.
    - shuffle: whether to shuffle the data.

    Returns:
    - train_loader: DataLoader for training set.
    - test_loader: DataLoader for test set.
    """
    size = len(subject_data_objects)
    idx = size // 2
    c = [0, idx, idx * 2]
    
    datal = subject_data_objects[c[0]:c[1]]
    datar = subject_data_objects[c[1]:c[2]]
    
    data = []
    for i in range(idx):
        data.extend([datal[i], datar[i]])
    
    size = len(data)
    test = data[size // 4:]
    train = data[:size // 4]
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, test_loader

######### Updated Model ########
class GATWithEntropy(nn.Module):
    def __init__(self, hidden_channels, heads):
        super(GATWithEntropy, self).__init__()

        # Define GAT convolution layers
        self.conv1 = GATv2Conv(22, hidden_channels, heads=heads, concat=False)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=False)
        self.conv3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=False)

        # Define GraphNorm layers
        self.gn1 = GraphNorm(hidden_channels * heads)
        self.gn2 = GraphNorm(hidden_channels * heads)
        self.gn3 = GraphNorm(hidden_channels * heads)
        
        #self.batch_norm = nn.BatchNorm1d(hidden_channels*heads)

        # Store attention scores
        self.attention_scores = []

    def forward(self, x, edge_index, batch):
        self.attention_scores = []  # Reset scores for each forward pass

        # Apply first GAT layer
        x, alpha1 = self.conv1(x, edge_index, return_attention_weights=True)
        #print(alpha1[0].shape)
        self.attention_scores.append(alpha1)
        x = F.relu(x)
        x = self.gn1(x, batch)  # Apply GraphNorm

        # Apply second GAT layer
        x, alpha2 = self.conv2(x, edge_index, return_attention_weights=True)
        self.attention_scores.append(alpha2)
        x = F.relu(x)
        x = self.gn2(x, batch)  # Apply GraphNorm

        # Apply third GAT layer
        x, alpha3 = self.conv3(x, edge_index, return_attention_weights=True)
        self.attention_scores.append(alpha3)
        x = self.gn3(x, batch)  # Apply GraphNorm

        # Global pooling
        x = global_mean_pool(x, batch)
        #x = self.batch_norm(x)

        return x

    def calculate_entropy(self, scores):
        """
        Calculate entropy for a given set of attention scores.
        """
        # Determine batch size dynamically
        batch_size = scores.size(0) // (22 * 22)  # Infer batch size from tensor shape
        #print("Batch size:", batch_size)  # Debug print
        
        # Reshape scores to [batch_size, 22, 22, heads]
        scores = scores.view(batch_size, 22, 22, -1)
        #print("Reshaped scores shape:", scores.shape)  # Debug print
        
        # Compute entropy for each electrode pair
        entropy_vals = -(scores * torch.log(scores + 1e-9)).sum(dim=-1)  # Sum along the heads dimension
        #print("Entropy shape after summing:", entropy_vals.shape)  # Debug print
        return entropy_vals
    
    def plot_entropy_heatmap(self, entropy_vals, batch_idx):
        """
        Plot heatmap of entropy for the given batch and entropy values.
        """
        # Convert to numpy for plotting
        entropy_vals_np = entropy_vals
    
        # Plotting the entropy heatmap for each graph in the batch
        plt.figure(figsize=(10, 8))
        sns.heatmap(entropy_vals_np, annot=True, cmap="viridis", fmt=".3f", linewidths=0.5)
        plt.title(f"Entropy Heatmap for Batch {batch_idx+1}")
        plt.xlabel("Electrode")
        plt.ylabel("Electrode")
        plt.show()
    
    def rank_connections_by_entropy(self):
        """
        Rank connections by attention entropy and visualize the last layer's entropy as a heatmap.
        """
        # Access the last layer's attention scores
        if not self.attention_scores:
            raise ValueError("Attention scores are not available.")
        
        last_layer_alpha = self.attention_scores[-1]  # Get the last layer's attention scores
        scores = last_layer_alpha[1]  # Extract attention scores from the tuple
    
        # Calculate entropy for the last layer
        entropy_vals = self.calculate_entropy(scores)  # Reuse the calculate_entropy method
        
        connection_ranks = []
        for batch_idx, batch_entropy in enumerate(entropy_vals):
            #print(f"Processing batch {batch_idx+1} out of {len(entropy_vals)}")
            
            # Plot entropy heatmap for the batch
            self.plot_entropy_heatmap(batch_entropy, batch_idx)
            
            # Rank the entropy values for each electrode pair
            ranked_connections = torch.argsort(batch_entropy.view(-1), descending=True)
            connection_ranks.append(ranked_connections)
        
        return connection_ranks


# Main loop for processing each subject's data
data_dir = 'C:/Users/uceerjp/Desktop/PhD/Multi-session Data/OG_Full_Data'
subject_numbers = [9]  # Example for one subject
torch.manual_seed(12345)

for subject_number in subject_numbers:
    # Initialize empty lists for data
    data_list = []
    subject_data = {}
    
    # Load data for this subject
    subject_data[f'S{subject_number}'] = load_subject_data(data_dir, subject_number)
    S1 = subject_data[f'S{subject_number}'][:, :]
    
    # Compute PLV and prepare labels
    plv, y = compute_plv(S1)
    
    # Create graphs from PLV
    graphs = create_graphs(plv)
    
    # Prepare adjacency matrices and edge indices
    adj, edge_indices = prepare_graph_data(graphs)
    
    # Create Data objects for PyTorch Geometric
    subject_data_objects = create_data_objects(adj, edge_indices, y)
    train_loader, test_loader = create_dataloaders(subject_data_objects, batch_size=32, shuffle=False)
    
    ### Initialize Model ###
    model = GATWithEntropy(hidden_channels=22, heads=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Variables to track the lowest loss epoch
    lowest_loss = float('inf')
    best_epoch = -1
    best_batch_entropy_vals = []
    
    ### Training Phase ###
    model.train()
    for epoch in range(1, 100):  # Train for 100 epochs
        epoch_loss = 0
        epoch_entropy_vals = []  # Store entropy values for all batches in the current epoch
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch)
            
            # Calculate loss and backpropagate
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()
            
            # Accumulate epoch loss
            epoch_loss += loss.item()
            
            # Calculate entropy for this batch and store
            if model.attention_scores:  # Ensure attention scores are available
                last_layer_alpha = model.attention_scores[-1]  # Last layer attention scores
                scores = last_layer_alpha[1]  # Extract attention scores
                entropy_vals = model.calculate_entropy(scores)  # Calculate entropy
                
                # Store entropy for this batch
                epoch_entropy_vals.append(entropy_vals.detach().cpu().numpy())
        
        # Compute average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}, Loss: {avg_epoch_loss:.4f}")
        
        # Optionally, save the best model based on the lowest loss
        if avg_epoch_loss < lowest_loss:
            lowest_loss = avg_epoch_loss
            best_epoch = epoch
            best_batch_entropy_vals = epoch_entropy_vals
    
    ### Testing Phase ###
    model.eval()  # Set model to evaluation mode
    test_entropy_vals = []  # Store entropy values for the test set
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch)
            
            # Calculate entropy for this batch
            if model.attention_scores:  # Ensure attention scores are available
                last_layer_alpha = model.attention_scores[-1]  # Last layer attention scores
                scores = last_layer_alpha[1]  # Extract attention scores
                entropy_vals = model.calculate_entropy(scores)  # Calculate entropy
                
                # Store entropy for this batch
                test_entropy_vals.append(entropy_vals.detach().cpu().numpy())
    
    # After training and testing, plot the entropy heatmaps for both training and testing
    # Plot training entropy heatmaps
    for batch_idx, entropy_vals in enumerate(best_batch_entropy_vals):
        model.plot_entropy_heatmap(np.mean(entropy_vals,axis=0), batch_idx)

    # Plot testing entropy heatmaps
    for batch_idx, entropy_vals in enumerate(test_entropy_vals):
        model.plot_entropy_heatmap(np.mean(entropy_vals,axis=0), batch_idx)


#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr  # Import pearsonr for p-value calculation

def create_voting_map(entropy_vals_list):
    # Create a 2D array to store the cumulative frequency of electrode pair selection
    # Assuming entropy_vals have shape (n_batches, 32, 22, 22), i.e., 32 samples per batch
    voting_map = np.zeros_like(entropy_vals_list[0][0])  # Shape will be (22, 22)

    # Iterate through all batches and all samples inside each batch
    for entropy_vals in entropy_vals_list:
        for sample_idx in range(entropy_vals.shape[0]):  # 32 samples in each batch
            # Sum entropy values for each electrode pair in the sample
            sample_entropy_vals = entropy_vals[sample_idx]
            # Increment the voting map based on the entropy values
            voting_map += sample_entropy_vals

    return voting_map

def plot_voting_map(voting_map, title):
    # Plot the final aggregated voting map as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(voting_map, annot=True, cmap="viridis", fmt=".3f", linewidths=0.5)
    plt.title(title)
    plt.show()

def plot_averaged_heatmaps(entropy_vals_list, title):
    # Create an array to store the averaged heatmaps
    averaged_map = np.zeros_like(entropy_vals_list[0][0])  # Shape (22, 22)

    # Iterate through all batches and compute the average heatmap for each batch
    for entropy_vals in entropy_vals_list:
        # Compute the average entropy across the batch
        avg_entropy_vals = np.mean(entropy_vals, axis=0)  # Averaging over samples in a batch
        averaged_map += avg_entropy_vals

    # Compute the final average heatmap across all batches
    averaged_map /= len(entropy_vals_list)
    
    # Plot the averaged heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(averaged_map, annot=True, cmap="viridis", fmt=".3f", linewidths=0.5)
    plt.title(title)
    plt.show()
    
    return averaged_map  # Return the averaged map for correlation calculation

def calculate_correlation(map1, map2):
    # Flatten the maps to 1D arrays
    map1_flat = map1.flatten()
    map2_flat = map2.flatten()
    
    # Calculate the Pearson correlation coefficient and p-value
    correlation, p_value = pearsonr(map1_flat, map2_flat)
    
    return correlation, p_value

# Assuming 'best_batch_entropy_vals' contains the entropy maps for the training batches
# and 'test_entropy_vals' contains the entropy maps for the testing batches

# Plot averaged heatmaps for training and testing batches and calculate correlation
train_avg_map = plot_averaged_heatmaps(best_batch_entropy_vals, "Averaged Training Heatmap")
test_avg_map = plot_averaged_heatmaps(test_entropy_vals, "Averaged Testing Heatmap")

# Calculate correlation and p-value between training and testing averaged maps
correlation, p_value = calculate_correlation(train_avg_map, test_avg_map)

# Print the correlation coefficient and p-value
print(f"Correlation between Averaged Training and Testing Heatmaps: {correlation:.4f}")
print(f"P-value: {p_value:.10f}")

