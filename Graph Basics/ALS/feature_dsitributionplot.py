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
import pandas as pd
from networkx.algorithms.community import greedy_modularity_communities
import seaborn as sns
from scipy.stats import entropy

#% Functions
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
    numElectrodes = subject_data['L'][0,1].shape[1]
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


def create_graphs(plv, threshold):
    
    graphs = []
    for i in range(plv.shape[2]):
        G = nx.Graph()
        G.add_nodes_from(range(plv.shape[0]))
        for u in range(plv.shape[0]):
            for v in range(plv.shape[0]):
                if u != v and plv[u, v, i] > threshold:
                    G.add_edge(u, v, weight=plv[u, v, i])
        #G = nx.normalized_laplacian_matrix(G) #Laplacian
        graphs.append(G)
    return graphs

# Function to compute feature distributions
def compute_feature_distribution(graphs):
    # List of electrode names
    electrode_names = [
        'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2'
    ]
    
    # Create a list to store the electrode pairs
    pairs = []
    
    # Get the indices of the lower triangle (excluding diagonal)
    lower_triangle_indices = np.tril_indices(19, k=-1)
    
    # Add only the lower triangle values as unique pairs
    for i, j in zip(*lower_triangle_indices):
        pair = (electrode_names[i], electrode_names[j])
        pairs.append(pair)
    
    # Convert pairs to a numpy array if needed
    pairs_array = np.array(pairs)
    
    # List to hold the feature distributions
    distributions = []
    
    for graph in graphs:
        adj_matrix = nx.to_numpy_array(graph, weight='weight')
        
        # Extract only the lower triangle of the matrix (excluding diagonal)
        lower_triangle_values = adj_matrix[lower_triangle_indices]
        
        # Flatten the lower triangle values (as a 1D array)
        flattened_values = lower_triangle_values.flatten()
        
        # Append the flattened values to the distributions list
        distributions.append(flattened_values)
    
    return np.array(distributions), pairs_array


# Function to split graphs and labels into sessions
def split_graphs_labels(graphs, y):
    # Assuming `graphs` is a list of graphs and `y` is a tensor of labels
    trials = len(graphs)  # Total number of trials
    
    # Split into Left and Right classes
    left_graphs = graphs[:trials // 2]  # First half are Left class
    right_graphs = graphs[trials // 2:]  # Second half are Right class
    left_labels = y[:trials // 2]  # Labels for Left class
    right_labels = y[trials // 2:]  # Labels for Right class
    
    # Split each class into four quarters (graphs)
    left_s1 = left_graphs[:len(left_graphs) // 4]
    left_s2 = left_graphs[len(left_graphs) // 4:len(left_graphs) // 2]
    left_s3 = left_graphs[len(left_graphs) // 2:3 * len(left_graphs) // 4]
    left_s4 = left_graphs[3 * len(left_graphs) // 4:]
    
    right_s1 = right_graphs[:len(right_graphs) // 4]
    right_s2 = right_graphs[len(right_graphs) // 4:len(right_graphs) // 2]
    right_s3 = right_graphs[len(right_graphs) // 2:3 * len(right_graphs) // 4]
    right_s4 = right_graphs[3 * len(right_graphs) // 4:]
    
    # Split each class into four quarters (labels)
    left_y1 = left_labels[:len(left_labels) // 4]
    left_y2 = left_labels[len(left_labels) // 4:len(left_labels) // 2]
    left_y3 = left_labels[len(left_labels) // 2:3 * len(left_labels) // 4]
    left_y4 = left_labels[3 * len(left_labels) // 4:]
    
    right_y1 = right_labels[:len(right_labels) // 4]
    right_y2 = right_labels[len(right_labels) // 4:len(right_labels) // 2]
    right_y3 = right_labels[len(right_labels) // 2:3 * len(right_labels) // 4]
    right_y4 = right_labels[3 * len(right_labels) // 4:]
    
    # Combine Left and Right quarters into sessions (graphs are just lists, no need to use torch.cat here)
    s1 = left_s1 + right_s1
    s2 = left_s2 + right_s2
    s3 = left_s3 + right_s3
    s4 = left_s4 + right_s4
    
    # Concatenate labels using torch.cat
    y1 = torch.cat((left_y1, right_y1), dim=0)
    y2 = torch.cat((left_y2, right_y2), dim=0)
    y3 = torch.cat((left_y3, right_y3), dim=0)
    y4 = torch.cat((left_y4, right_y4), dim=0)
    
    return s1, s2, s3, s4, y1, y2, y3, y4

# % Preparing Data
data_dir = 'C:/Users/uceerjp/Desktop/PhD/Multi-session Data/OG_Full_Data'
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]
#subject_numbers = [1]
subject_data = {}
output_dir = "Feature_Distribution_Plots"
os.makedirs(output_dir, exist_ok=True)  # Create a directory for saving plots

for subject_number in subject_numbers:
    mat_fname = pjoin(data_dir, f'OGFS{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_data[f'S{subject_number}'] = mat_contents[f'Subject{subject_number}']
    S1 = subject_data[f'S{subject_number}'][:,:]
    plv, y = compute_plv(S1)
    plv = plv[:19, :19, :]  # Ensure the shape matches expected input
    graphs = create_graphs(plv, threshold=0.0)

    s1, s2, s3, s4, y1, y2, y3, y4 = split_graphs_labels(graphs, y)
    
    # Compute distributions for each session
    session_distributions = {
        "session1": compute_feature_distribution(s1),
        "session2": compute_feature_distribution(s2),
        "session3": compute_feature_distribution(s3),
        "session4": compute_feature_distribution(s4),
    }
    session_labels = {
        "session1": y1,
        "session2": y2,
        "session3": y3,
        "session4": y4,
    }
    
    # Create subject-specific directory
    subject_dir = os.path.join(output_dir, f"Subject{subject_number}")
    os.makedirs(subject_dir, exist_ok=True)
    
    # List of electrode names
    electrode_names = [
        'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2'
    ]
    
    # Create session directories within the subject folder
    for session_name, (distributions, electrode_pairs) in session_distributions.items():
        session_label = session_labels[session_name].numpy()
        
        if len(session_label) != len(distributions):
            print(f"Error: The number of labels for {session_name} does not match the number of distributions.")
            continue
        
        left_distributions = distributions[:40, :]
        right_distributions = distributions[40:, :]
        
        # Calculate the averages for left and right
        avg_left = np.mean(left_distributions, axis=0) if left_distributions.size > 0 else np.zeros(distributions.shape[1])
        avg_right = np.mean(right_distributions, axis=0) if right_distributions.size > 0 else np.zeros(distributions.shape[1])
        
        # Plot the averages for the session on the same plot
        plt.figure(figsize=(12, 8))
        
        # Plot Left class average
        plt.plot(avg_left, label="Left Class Average", color='blue')
        
        # Plot Right class average
        plt.plot(avg_right, label="Right Class Average", color='red')
        
        # Title and labels
        plt.title(f"Average Feature Distributions: Subject:{subject_number},{session_name}")
        plt.xlabel("Electrode Pair")
        plt.ylabel("Average PLV")
        # Define a step for tick spacing (e.g., every 2nd electrode pair)
        tick_step = 2  # Change this value to adjust spacing between ticks
        
        # Set xticks with the specified step for spacing
        plt.xticks(ticks=np.arange(0, len(electrode_pairs), tick_step), 
                   labels=[f"{pair[0]}-{pair[1]}" for pair in electrode_pairs][::tick_step], 
                   rotation=90)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Show the plot
        plt.show()
        
        # Create session-specific directory
        session_dir = os.path.join(subject_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)
        
        # Plot and save distributions for each trial in the session
        for trial_idx, prob_dist in enumerate(distributions):
            label = session_label[trial_idx].item()  # Get the label for this trial
            label_name = "Left" if label == 0 else "Right"  # Replace with your class names if different
            
            plt.figure(figsize=(12, 8))
            plt.plot(prob_dist, label=f'Trial {trial_idx + 1}')
            plt.title(
                f"Feature Distribution: Subject {subject_number} - {session_name} - Trial {trial_idx + 1} (Label: {label_name})"
            )
            plt.xlabel("Electrode Pair (e.g., FP1-FP2)")
            plt.ylabel("PLV")
            # Define a step for tick spacing (e.g., every 2nd electrode pair)
            tick_step = 2  # Change this value to adjust spacing between ticks
            # Set xticks with the specified step for spacing
            plt.xticks(ticks=np.arange(0, len(electrode_pairs), tick_step), 
           labels=[f"{pair[0]}-{pair[1]}" for pair in electrode_pairs][::tick_step], 
           rotation=90)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(session_dir, f"Trial_{trial_idx + 1}_Label_{label_name}.png")
            plt.savefig(plot_path)
            plt.close()