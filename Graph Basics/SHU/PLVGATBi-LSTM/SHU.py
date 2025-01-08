# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:26:26 2024
Rishan Patel, UCL, Bioelectronics Group.



https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
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
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GAT, GraphNorm
from torch_geometric.nn import global_mean_pool
from torch import nn
import pickle
from progressbar import progressbar
import os
import scipy
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import mne
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from mne.decoding import CSP
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
from sklearn.metrics import accuracy_score
from torch_geometric.data import Data
import time
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.seed import seed_everything


def get_model_memory(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.element_size() * p.numel() for p in model.parameters()) + \
        sum(b.element_size() * b.numel() for b in model.buffers())
    return print(total_params, total_size / (1024 ** 2))  # Return memory in MB



## Functions
def split_data_by_label(data_by_subject):
    data_split = {}

    for subject, data_dict in data_by_subject.items():
        # Extract data and labels
        data = data_dict['data']       # Shape: Trials x Channels x Samples
        labels = data_dict['labels']   # Shape: 1 x Trials (or Trials, depending on how it's stored)
        
        # Ensure labels are a 1D array if they are stored with shape (1, Trials)
        if labels.ndim == 2:
            labels = labels.flatten()
        
        # Initialize lists for Left and Right motor imagery
        data_L = []
        data_R = []
        
        # Iterate over trials and separate based on label
        for i in range(data.shape[0]):
            if labels[i] == 1:
                data_L.append(data[i])
            elif labels[i] == 2:
                data_R.append(data[i])
        
        # Convert lists to numpy arrays
        data_L = np.array(data_L)
        data_R = np.array(data_R)
        
        # Store split data in the dictionary
        data_split[subject] = {'L': data_L, 'R': data_R}
    
    return data_split

def bandpass_filter_trials(data_split, low_freq, high_freq, sfreq):
    filtered_data_split = {}

    # Design the bandpass filter
    nyquist = 0.5 * sfreq
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(N=4, Wn=[low, high], btype='band')

    for subject in data_split:
        subject_data = data_split[subject]
        filtered_subject_data = {}

        for direction in ['L', 'R']:
            trials = subject_data[direction]  # Shape: (trials, channels, samples)
            filtered_trials = []

            for trial in range(trials.shape[0]):
                trial_data = trials[trial, :, :]  # Shape: (channels, samples)

                # Apply bandpass filter to each channel
                filtered_trial_data = np.zeros_like(trial_data)
                for ch in range(trial_data.shape[0]):
                    filtered_trial_data[ch, :] = filtfilt(b, a, trial_data[ch, :])
                
                filtered_trials.append(filtered_trial_data)

            # Convert the list of filtered trials back to a NumPy array
            filtered_subject_data[direction] = np.array(filtered_trials)

        filtered_data_split[subject] = filtered_subject_data

    return filtered_data_split

def split_into_folds(data, n_splits=5):
    """Splits data into n_splits causal folds."""
    n_trials = data.shape[0]
    trials_per_fold = n_trials // n_splits
    
    # Initialize fold lists
    folds = [None] * n_splits
    
    for i in range(n_splits):
        start_index = i * trials_per_fold
        if i == n_splits - 1:
            end_index = n_trials  # Last fold gets all remaining trials
        else:
            end_index = (i + 1) * trials_per_fold
        
        folds[i] = data[start_index:end_index]
    
    return folds

def prepare_folds_for_subjects(filtered_data_split, n_splits=5):
    """Splits filtered data into causal folds for each subject and direction."""
    fold_data = {}

    for subject in filtered_data_split:
        subject_data = filtered_data_split[subject]
        fold_data[subject] = {}
        
        for direction in ['L', 'R']:
            trials = subject_data[direction]  # Shape: (trials, channels, samples)
            folds = split_into_folds(trials, n_splits=n_splits)
            
            fold_data[subject][direction] = folds
    
    return fold_data

def plvfcn(eegData):
    numElectrodes = eegData.shape[0]
    numTimeSteps = eegData.shape[1]
    plvMatrix = np.zeros((numElectrodes, numElectrodes))
    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            phase1 = np.angle(sig.hilbert(eegData[electrode1, :]))
            phase2 = np.angle(sig.hilbert(eegData[electrode2, :]))
            phase_difference = phase2 - phase1
            plv = np.abs(np.sum(np.exp(1j * phase_difference)) / numTimeSteps)
            plvMatrix[electrode1, electrode2] = plv
            plvMatrix[electrode2, electrode1] = plv
    return plvMatrix


# Function to create graphs and compute adjacency matrices for PLV matrices
def create_graphs_and_edges(plv_matrices, threshold):
    graphs = []
    adj_matrices = np.zeros([plv_matrices.shape[0], plv_matrices.shape[0], plv_matrices.shape[2]])  # (Electrodes, Electrodes, Trials)
    edge_indices = []
    
    for i in range(plv_matrices.shape[2]):
        G = nx.Graph()
        G.add_nodes_from(range(plv_matrices.shape[0]))  # Nodes represent electrodes
        
        # Initialize lists for storing adjacency matrix and edge indices
        source_nodes = []
        target_nodes = []
        
        # Iterate over electrode pairs to construct the graph and adjacency matrix
        for u in range(plv_matrices.shape[0]):
            for v in range(u + 1, plv_matrices.shape[0]):  # Avoid duplicate edges
                if plv_matrices[u, v, i] > threshold:
                    # Add edge to graph
                    G.add_edge(u, v, weight=plv_matrices[u, v, i])
                    
                    # Store the edge in adjacency matrix (symmetric)
                    adj_matrices[u, v, i] = plv_matrices[u, v, i]
                    adj_matrices[v, u, i] = plv_matrices[u, v, i]
                    
                    # Store source and target nodes for edge indices
                    source_nodes.append(u)
                    target_nodes.append(v)
        
        # Convert adjacency matrix and graphs
        graphs.append(G)
        
        # Convert the lists to a LongTensor for edge index
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        edge_indices.append(edge_index)
    
    # Convert adjacency matrices to torch tensors
    adj_matrices = torch.tensor(adj_matrices, dtype=torch.float32)
    
    # Stack edge indices for all trials (session-wise)
    edge_indices = torch.stack(edge_indices, dim=-1)
    
    return adj_matrices, edge_indices, graphs


# Function to compute PLV matrices for all trials in all sessions of a subject
def compute_plv(subject_data):
    session_plv_data = {}  # To store PLV matrices, labels, and graphs for each session
    
    # Iterate over each session in the subject's data
    for session_id, session_data in subject_data.items():
        data = session_data['data']  # Shape: (Trials, Channels, TimeSteps)
        labels = session_data['label']  # Shape: (Trials,)
        
        numTrials, numElectrodes, _ = data.shape
        plv_matrices = np.zeros((numElectrodes, numElectrodes, numTrials))
        
        # Compute PLV for each trial in the session
        for trial_idx in range(numTrials):
            eeg_trial_data = data[trial_idx]  # Shape: (Channels, TimeSteps)
            plv_matrices[:, :, trial_idx] = plvfcn(eeg_trial_data)
        
        # Convert labels to torch tensor
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Create graphs, adjacency matrices, and edge indices
        adj_matrices, edge_indices, graphs = create_graphs_and_edges(plv_matrices, threshold=0)
        
        # Store session-level data for PLV matrices, labels, and graphs
        session_plv_data[session_id] = {
            'plv_matrices': plv_matrices,   # Shape: (Electrodes, Electrodes, Trials)
            'labels': label_tensor,         # Shape: (Trials,)
            'adj_matrices': adj_matrices,   # Shape: (Electrodes, Electrodes, Trials)
            'edge_indices': edge_indices,   # Shape: (2, Edges, Trials)
            'graphs': graphs                # List of graphs for each trial
        }
    
    return session_plv_data


def create_loaders(all_session_data, session_mask, num_folds=5):
    """
    Creates train and test DataLoaders based on the provided session mask and fold configuration.

    Parameters:
        all_session_data (dict): Dictionary containing session data.
        session_mask (list): Binary list indicating which sessions to use for training (1) and testing (0).
        num_folds (int): Number of folds to split the data into for cross-validation.

    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
    """
    data_list = []

    # Prepare data for each session and trial
    for session_idx in range(len(session_mask)):
        session = all_session_data[session_idx]
        for i in range(np.size(session['plv_matrices'], 2)):
            data_point = Data(
                x=session['adj_matrices'][:, :, i],
                edge_index=session['edge_indices'][:, :, i],
                y=session['labels'][i]
            )
            data_list.append((data_point, session_idx))  # Store data_point and its session index

    # Split data into train and test based on session_mask
    train_data = []
    test_data = []

    for data_point, session_idx in data_list:
        if session_mask[session_idx] == 1:
            train_data.append(data_point)
        else:
            test_data.append(data_point)

    # Split data into N folds for cross-validation (if needed)
    fold_size = len(train_data) // num_folds
    folds = [train_data[i * fold_size:(i + 1) * fold_size] for i in range(num_folds)]

    # Split the data into train_loader and test_loader
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    return train_loader, test_loader


class GATLSTM(nn.Module):
    def __init__(self, in_channels, gat_out_channels, lstm_hidden_size, num_classes, num_heads, num_lstm_layers):
        super(GATLSTM, self).__init__()

        # GAT module
        self.gat = GATv2Conv(in_channels, gat_out_channels, heads=num_heads, concat=False)

        # LSTM module with bidirectional=True
        self.lstm = nn.LSTM(input_size=gat_out_channels, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=num_lstm_layers, 
                            batch_first=True, dropout=0.5, bidirectional=True)

        # Fully connected layer - adjust input size for bidirectional LSTM (hidden_size * 2)
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)  # multiplied by 2 for bidirectional LSTM

    def forward(self, batch):
        x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch

        # Apply GAT to node features
        node_embeddings = self.gat(x, edge_index)

        # Pool node embeddings to get graph-level embeddings
        graph_embeddings = global_mean_pool(node_embeddings, batch_index)

        # Reshape for LSTM input (batch_size, sequence_length=1, gat_out_channels)
        graph_embeddings = graph_embeddings.unsqueeze(1)

        # Process through LSTM
        lstm_output, (h_n, c_n) = self.lstm(graph_embeddings)

        # h_n has the shape (num_layers * num_directions, batch_size, lstm_hidden_size)
        # For bidirectional LSTM, we concatenate the last hidden states from both directions
        # So, we need to select the final hidden state (from the last LSTM layer and the concatenated bidirectional output)
        final_hidden_state = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=-1)

        # Use the final hidden state for classification
        logits = self.fc(final_hidden_state)
        return logits


def train_and_evaluate(train_loader, test_loader):
    # Define fixed hyperparameters
    gat_out_channels = 32
    lstm_hidden_size = 64
    num_heads = 1
    num_lstm_layers = 1
    lr = 1e-3

    # Initialize the model
    model = GATLSTM(
        in_channels=32, 
        gat_out_channels=gat_out_channels,
        lstm_hidden_size=lstm_hidden_size, 
        num_classes=2,
        num_heads=num_heads, 
        num_lstm_layers=num_lstm_layers
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    get_model_memory(model)
    
    # Train and evaluate the model
    num_epochs = 200
    best_test_accuracy = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            loss.backward()
            optimizer.step()

        # Evaluation phase
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(batch)
                test_correct += (outputs.argmax(dim=1) == batch.y).sum().item()

        # Calculate test accuracy
        test_accuracy = test_correct / len(test_loader.dataset) * 100
        best_test_accuracy = max(best_test_accuracy, test_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {test_accuracy:.2f}%")

    print(f"Best Test Accuracy: {best_test_accuracy:.2f}%")
    return best_test_accuracy


# Define the directory containing your .mat files
directory = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\SHU Dataset\MatFiles'
#directory = r'/home/uceerjp/Multi-sessionData/SHU Dataset/MatFiles/'
seed_everything(12345) # Seed for everything

# Define constants
fs = 250  # Sampling frequency
channels = np.array([
    'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5',
    'FC6', 'CZ', 'C3', 'C4', 'T3', 'T4', 'A1', 'A2', 'CP1', 'CP2',
    'CP5', 'CP6', 'PZ', 'P3', 'P4', 'T5', 'T6', 'PO3', 'PO4', 'OZ',
    'O1', 'O2'
])

# Functions assumed to be defined: split_data_by_label, bandpass_filter_trials, prepare_folds_for_subjects, compute_plv, create_loaders, train_and_evaluate

# Initialize results storage
accuracy_history = {}

# Group files by subject
subject_files = {}
for filename in os.listdir(directory):
    if filename.endswith('.mat'):
        subject_id = filename.split('_')[0]  # e.g., 'sub-001'
        if subject_id not in subject_files:
            subject_files[subject_id] = []
        subject_files[subject_id].append(os.path.join(directory, filename))

# Process data for each subject
for subject_id, file_list in subject_files.items():
    print(f"Processing Subject: {subject_id}")
    subject_data = {'data': [], 'labels': []}

    # Load and concatenate data for the subject
    for file_path in file_list:
        mat_data = loadmat(file_path)
        data = mat_data['data']  # Assuming key 'data'
        labels = mat_data['labels']  # Assuming key 'labels'
        subject_data['data'].append(data)
        subject_data['labels'].append(labels)

    # Concatenate data and labels
    concatenated_data = np.concatenate(subject_data['data'], axis=0)
    concatenated_labels = np.concatenate(subject_data['labels'], axis=1)
    subject_data = {'data': concatenated_data, 'labels': concatenated_labels}

    # Process data: Split, filter, and prepare folds
    data_split = split_data_by_label({subject_id: subject_data})
    filtered_data_split = bandpass_filter_trials(data_split, low_freq=8, high_freq=30, sfreq=fs)
    fold_data = prepare_folds_for_subjects(filtered_data_split)

    # Merge fold data for classification
    merged_fold_data = {}
    for fold_id in range(5):  # Assuming 5 folds per session
        left_data = fold_data[subject_id]['L'][fold_id]
        right_data = fold_data[subject_id]['R'][fold_id]

        combined_data = np.concatenate((left_data, right_data), axis=0)
        left_labels = np.zeros(left_data.shape[0], dtype=int)
        right_labels = np.ones(right_data.shape[0], dtype=int)
        combined_labels = np.concatenate((left_labels, right_labels), axis=0)

        merged_fold_data[fold_id] = {
            'data': combined_data,
            'label': combined_labels
        }

    # Compute PLV matrices and prepare graph data
    subject_plv_data = compute_plv(merged_fold_data)
    all_session_data = subject_plv_data
    num_sessions = len(all_session_data)

    # Perform Leave-One-Session-Out Cross-Validation
    subject_accuracy = []
    for test_session_idx in range(num_sessions):
        # Create session mask: 1 for train, 0 for test
        session_mask = [1 if i != test_session_idx else 0 for i in range(num_sessions)]

        # Create train and test loaders
        train_loader, test_loader = create_loaders(all_session_data, session_mask)

        # Train and evaluate for the current test session
        accuracy = train_and_evaluate(train_loader, test_loader)
        subject_accuracy.append(accuracy)

        print(f"Subject {subject_id}, Test Session {test_session_idx}: Accuracy: {accuracy:.2f}%")

    # Store average accuracy across all sessions for the subject
    accuracy_history[subject_id] = {
        'average_accuracy': np.mean(subject_accuracy),
        'session_accuracies': subject_accuracy
    }

    print(f"Subject {subject_id}: Average Accuracy: {accuracy_history[subject_id]['average_accuracy']:.2f}%")

print("Processing complete.")


#%%


# Calculate average and std per subject
results = {}
for subject, data in accuracy_history.items():
    session_accuracies = data['session_accuracies']
    avg = np.mean(session_accuracies)
    std = np.std(session_accuracies)
    results[subject] = {'average_accuracy': avg, 'std_accuracy': std}

# Display results
for subject, stats in results.items():
    print(f"{subject}: Average Accuracy = {stats['average_accuracy']:.2f}, STD = {stats['std_accuracy']:.2f}")
