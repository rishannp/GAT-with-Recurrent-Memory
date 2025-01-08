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

def prepare_subject_data(S1, threshold=0.1, num_folds=2):
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

    # Prepare training and testing sets
    for train_fold_idx in range(num_folds):
        if train_fold_idx != 0:  # Skip other folds as training data
            continue

        for test_fold_idx in range(num_folds):
            if test_fold_idx == train_fold_idx or test_fold_idx == 0:  # Skip if testing fold is the same as training fold or fold 1
                continue

            train = folds[train_fold_idx]
            test = folds[test_fold_idx]

            # Create DataLoader for train and test sets
            train_loader = DataLoader(train, batch_size=32, shuffle=False)
            test_loader = DataLoader(test, batch_size=32, shuffle=False)

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
        out, w3, att3 = model(data.x, data.edge_index, data.batch)

        # Append the attention scores for the current batch to the list
        all_att.append(att3[1])
        all_edge.append(att3[0])
        all_w.append(w3)

        # Calculate loss
        loss = criterion(out, data.y)
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

    return correct / len(loader.dataset), inference_time, all_w, all_att, all_edge


# % Preparing Data
data_dir = 'C:/Users/uceerjp/Desktop/PhD/Multi-session Data/OG_Full_Data'
#subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]
subject_numbers = [39]
subject_data = {}
all_subjects_accuracies = {}

# Initialize fold_accuracies outside the subject loop
fold_accuracies = []

# Loop through subjects
for subject_number in subject_numbers:
    mat_fname = pjoin(data_dir, f'OGFS{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    S1 = mat_contents[f'Subject{subject_number}'][:, :]

    # Prepare train and test loaders for this subject
    train_loader, test_loader = prepare_subject_data(S1)

    # Initialize the model, optimizer, and loss function
    model = GAT(hidden_channels=22, heads=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    optimal_fold_acc = 0  # Store only the accuracy (float)
    fold_test_accuracies = []
    fold_weights = {}
    fold_att = {}
    train_times = []
    inference_times = []
    ram = []

    start_train_time = time.time()  # Start timing for the entire training process

    for epoch in range(1, 501):  # Note: Range is 1 to 500 (inclusive)
        epoch_train_time, train_acc, trainw3, trainatt3, trainedge3 = train()
        train_times.append(epoch_train_time)
        
        test_acc, inference_time, testw3, testatt3, testedge3 = test(test_loader)
        inference_times.append(inference_time)

        ram_usage = get_ram_usage()

        fold_test_accuracies.append(test_acc)
        
        print(f"Epoch {epoch}: Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")


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

# Saving the results
save_file_path = r"C:\path\to\your\directory\all_subjects_accuracies.pkl"
with open(save_file_path, 'wb') as f:
    pickle.dump(all_subjects_accuracies, f)

print("Dictionary saved successfully.")

# %% load data
import pickle
# Define the path to the pickle file
load_file_path = r"C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Understanding Stability in Graphical Representation\GAT\adjAsX\Session 1 v X Pairs\ALS - V2\Explain\all_subjects_accuracies.pkl"

# Load the dictionary from the pickle file
with open(load_file_path, 'rb') as f:
    all_subjects_accuracies = pickle.load(f)

print("Dictionary loaded successfully.")

#%% Train Weights plotting

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Define your electrode labels (22 electrodes)
electrode_labels = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7",
                    "C3", "CZ", "C4", "T8", "P7", "P3", "PZ", "P4",
                    "P8", "O1", "O2"]

# Base directory to save the plots
base_save_path = r"C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Understanding Stability in Graphical Representation\Plots\PLV_weights\ALS"

# Iterate through each subject
for subject, train_test_pairs in all_subjects_accuracies.items():
    for pair_idx, pair_data in enumerate(train_test_pairs):
        # Extract the best weights (w3) for this train/test pair
        best_weights = pair_data['best_weights']
        w3 = best_weights['w3']  # The weights array of shape (Trials, 22)

        # Calculate number of trials
        num_trials = w3.shape[0]

        # Define the labels for trials (Left/Right)
        labels = ['Left' if i % 2 == 0 else 'Right' for i in range(num_trials)]

        # Convert 'Left'/'Right' labels to numeric values (0 for Left, 1 for Right)
        label_map = {'Left': 0, 'Right': 1}
        numeric_labels = [label_map[label] for label in labels]

        # Create directories for the subject and fold pair
        subject_save_path = os.path.join(base_save_path, subject)
        os.makedirs(subject_save_path, exist_ok=True)
        
        fold_save_path = os.path.join(subject_save_path, f"TrainTestPair_{pair_idx}")
        os.makedirs(fold_save_path, exist_ok=True)

        # Create folder for storing the heatmaps and PCA plots
        heatmap_folder = os.path.join(fold_save_path, "Heatmaps")
        os.makedirs(heatmap_folder, exist_ok=True)

        # Plotting heatmap of the weights for each trial (electrode weights)
        plt.figure(figsize=(16, 16))  # Increase figure size for x-axis length
        sns.heatmap(
            w3,
            cmap="RdBu",
            annot=False,
            cbar=True,
            xticklabels=electrode_labels,
            yticklabels=numeric_labels,
            square=True,
            cbar_kws={'label': 'Weight Value'}  # Label for the color bar
        )
        
        # Rotate axis labels to prevent overlap
        plt.xticks(rotation=90, ha='right', fontsize=10)
        plt.yticks(ticks=range(len(numeric_labels)), labels=numeric_labels, rotation=0, fontsize=10)

        plt.title(f"Best Weights for {subject} TrainTestPair {pair_idx}", fontsize=16)
        plt.xlabel("Electrodes", fontsize=12)
        plt.ylabel("Trials (0=Left, 1=Right)", fontsize=12)
        
        # Save the heatmap plot
        heatmap_save_file = os.path.join(heatmap_folder, f"Best_Weights_Heatmap_{subject}_Pair{pair_idx}.png")
        plt.savefig(heatmap_save_file, dpi=300, bbox_inches='tight')
        plt.close()

print("All plots have been saved in their respective folders.")

#%% Understanding through weights which channels are important for each class

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define your electrode labels (22 electrodes)
electrode_labels = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7",
                    "C3", "CZ", "C4", "T8", "P7", "P3", "PZ", "P4",
                    "P8", "O1", "O2","","",""]

# Base directory to save the plots
base_save_path = r"C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Understanding Stability in Graphical Representation\Plots\PLV_weights\ALS"

# Iterate through each subject
for subject, train_test_pairs in all_subjects_accuracies.items():
    for pair_idx, pair_data in enumerate(train_test_pairs):
        # Extract the best weights (w3) for this train/test pair
        best_weights = pair_data['best_weights']
        w3 = best_weights['w3']  # The weights array of shape (Trials, 22)

        # Debugging shapes
        print(f"w3 shape: {w3.shape}")
        print(f"electrode_labels length: {len(electrode_labels)}")
        
        # Ensure w3 dimensions match electrodes and trials
        num_trials, num_electrodes = w3.shape
        assert num_electrodes == len(electrode_labels), "Mismatch between electrode labels and w3 columns."
        
        # Define the labels for trials (Left/Right)
        labels = ['Left' if i % 2 == 0 else 'Right' for i in range(num_trials)]
        label_map = {'Left': 0, 'Right': 1}
        numeric_labels = [label_map[label] for label in labels]

        # Create directories for the subject and fold pair
        subject_save_path = os.path.join(base_save_path, subject)
        os.makedirs(subject_save_path, exist_ok=True)
        
        fold_save_path = os.path.join(subject_save_path, f"TrainTestPair_{pair_idx}")
        os.makedirs(fold_save_path, exist_ok=True)

        # Create folder for storing the heatmaps and bar plots
        heatmap_folder = os.path.join(fold_save_path, "Heatmaps")
        barplot_folder = os.path.join(fold_save_path, "BarPlots")
        os.makedirs(heatmap_folder, exist_ok=True)
        os.makedirs(barplot_folder, exist_ok=True)

        # Plotting heatmap of the weights for each trial (electrode weights)
        plt.figure(figsize=(16, 12))  # Adjust figure size for better visibility
        sns.heatmap(
            w3,
            cmap="RdBu",
            annot=False,
            cbar=True,
            xticklabels=electrode_labels,
            yticklabels=numeric_labels,
            square=False,  # Allow flexibility in shape
            cbar_kws={'label': 'Weight Value'}  # Label for the color bar
        )
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.title(f"Best Weights for {subject} TrainTestPair {pair_idx}", fontsize=16)
        plt.xlabel("Electrodes", fontsize=12)
        plt.ylabel("Trials (0=Left, 1=Right)", fontsize=12)
        
        # Save the heatmap plot
        heatmap_save_file = os.path.join(heatmap_folder, f"Best_Weights_Heatmap_{subject}_Pair{pair_idx}.png")
        plt.savefig(heatmap_save_file, dpi=300, bbox_inches='tight')
        plt.close()

        # Calculate average weights for each class
        class_0_indices = [i for i, label in enumerate(numeric_labels) if label == 0]
        class_1_indices = [i for i, label in enumerate(numeric_labels) if label == 1]
        avg_weights_class_0 = np.mean(w3[class_0_indices], axis=0)
        avg_weights_class_1 = np.mean(w3[class_1_indices], axis=0)

        # Bar plot to visualize average weights for each electrode
        plt.figure(figsize=(14, 8))
        x = np.arange(len(electrode_labels))
        width = 0.35

        plt.bar(x - width / 2, avg_weights_class_0, width, label='Class 0 (Left)', color='blue')
        plt.bar(x + width / 2, avg_weights_class_1, width, label='Class 1 (Right)', color='red')
        plt.xticks(x, electrode_labels, rotation=45, ha='right', fontsize=10)
        plt.ylabel("Average Weight Value", fontsize=12)
        plt.xlabel("Electrodes", fontsize=12)
        plt.title(f"Average Electrode Weights for {subject} TrainTestPair {pair_idx}", fontsize=16)
        plt.legend(fontsize=12)

        # Save the bar plot
        barplot_save_file = os.path.join(barplot_folder, f"Average_Weights_BarPlot_{subject}_Pair{pair_idx}.png")
        plt.savefig(barplot_save_file, dpi=300, bbox_inches='tight')
        plt.close()

print("All plots have been saved in their respective folders.")

# Prepare data
data = []
for subject, folds in all_subjects_accuracies.items():
    for fold_data in folds:
        train_test_pair = fold_data['fold_name']
        # Using 'optimal' accuracy here, you can change it to 'mean', 'high', or 'low'
        accuracy = fold_data['optimal']
        data.append({
            'Subject': subject,  # E.g., 'S1', 'S2', etc.
            'Train-Test Pair': train_test_pair,
            'Accuracy': accuracy
        })

df = pd.DataFrame(data)


plt.figure(figsize=(14, 8))
sns.boxplot(x='Subject', y='Accuracy', data=df)
plt.title('Accuracy Distribution Across Subjects')
plt.xlabel('Subject')
plt.ylabel('Accuracy')
plt.xticks(rotation=90)  # Rotate x-axis labels if needed
plt.tight_layout()
plt.show()

# Prepare data for heatmap
heatmap_data = df.pivot(
    index='Subject', columns='Train-Test Pair', values='Accuracy')

plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title('Accuracy Heatmap Across Subjects and Train-Test Pairs')
plt.xlabel('Train-Test Pair')
plt.ylabel('Subject')
plt.xticks(ticks=range(len(heatmap_data.columns)),
           labels=heatmap_data.columns, rotation=90)
plt.yticks(ticks=range(len(heatmap_data.index)),
           labels=heatmap_data.index, rotation=0)
plt.tight_layout()
plt.show()


# % Average the Accuracies

# Dictionary to hold the calculated average statistics for each subject
average_stats = {}

# Iterate over each subject in the all_subjects_accuracies dictionary
for subject, folds_list in all_subjects_accuracies.items():
    total_mean = 0
    total_high = 0
    total_low = 0
    total_optimal = 0
    num_folds = len(folds_list)

    # Iterate over each fold (dict) in the subject's list of folds
    for fold_dict in folds_list:
        total_mean += fold_dict['mean']
        total_high += fold_dict['high']
        total_low += fold_dict['low']
        total_optimal += fold_dict['optimal']

    # Calculate averages for this subject
    average_stats[subject] = {

        'Average Mean': total_mean / num_folds,
        'Average High': total_high / num_folds,
        'Average Low': total_low / num_folds,
        'Average Optimal': total_optimal / num_folds
    }

# Example: Print average statistics for each subject
for subject, averages in average_stats.items():
    print(f"Subject {subject}:")
    print(f"  Average Mean: {averages['Average Mean']:.4f}")
    print(f"  Average High: {averages['Average High']:.4f}")
    print(f"  Average Low: {averages['Average Low']:.4f}")
    print(f"  Average Optimal: {averages['Average Optimal']:.4f}")


# Initialize accumulators
total_ram_usage = 0
total_train_time = 0
total_inference_time = 0
total_folds = 0

# Iterate over all subjects and folds
for subject, folds in all_subjects_accuracies.items():
    for metrics in folds:
        total_ram_usage += metrics['ram']
        total_train_time += metrics['total_train_time']
        total_inference_time += metrics['avg_inference_time']
        total_folds += 1

# Calculate averages
avg_ram_usage = total_ram_usage / total_folds
avg_train_time = total_train_time / total_folds
avg_inference_time = total_inference_time / total_folds

# Print averages
print(f"Average RAM Usage: {avg_ram_usage:.2f} GB")
print(f"Average Training Time: {avg_train_time:.5f} seconds")
print(f"Average Inference Time: {avg_inference_time:.5f} seconds")


#%% Attention plotting
# Define the base save path for plots
save_path = r"C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Understanding Stability in Graphical Representation\GAT\adjAsX\Session 1 v X Pairs\ALS - V2\Explain"

# Electrode names for X and Y axes
electrode_labels = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7",
                    "C3", "CZ", "C4", "T8", "P7", "P3", "PZ", "P4",
                    "P8", "O1", "O2"]

# Iterate through each subject in the dictionary
for subject, train_test_pairs in all_subjects_accuracies.items():
    for pair_idx, pair_data in enumerate(train_test_pairs):
        # Extract the best attention scores (att3) for this train/test pair
        best_attention_scores = pair_data['best_attention_scores']
        att3 = best_attention_scores['att3']  # Attention scores (numpy array)

        # Calculate number of trials and reshape att3 to [22, 22, num_trials]
        num_trials = att3.shape[0] // (22 * 22)  # Calculate number of full trials

        # Reshape att3 to [22, 22, num_trials]
        reshaped_att3 = np.reshape(att3[:num_trials * 22 * 22], [22, 22, num_trials])

        # Create directories for saving plots
        layer_save_path = os.path.join(
            save_path, subject, f"TrainTestPair_{pair_idx}", "Layer_att3")
        os.makedirs(layer_save_path, exist_ok=True)

        # Loop through each trial to visualize the attention scores
        for trial_idx in range(num_trials):
            # Extract attention scores for this trial
            trial_scores = reshaped_att3[:, :, trial_idx]

            # Plot the heatmap with fixed vmin and vmax
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                trial_scores,
                cmap="RdBu",
                annot=False,
                cbar=True,
                vmin=0, vmax=1,  # Fix the color range to 0-1
                xticklabels=electrode_labels,
                yticklabels=electrode_labels,
                square=True
            )
            plt.title(
                f"{subject} TrainTestPair {pair_idx} Layer att3 Trial {trial_idx}", fontsize=14)
            plt.xlabel("Electrodes", fontsize=12)
            plt.ylabel("Electrodes", fontsize=12)

            # Save the plot
            save_file = os.path.join(
                layer_save_path, f"Trial_{trial_idx}.png")
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            plt.close()

print("All heatmaps have been plotted and saved.")
