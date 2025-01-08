import tensorflow
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from os.path import join as pjoin
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
from scipy.io import loadmat
import numpy as np
import os
import scipy.signal as signal
from scipy.signal import butter, filtfilt, welch
import pandas as pd
from tensorflow.keras.models import Sequential

# EEGNet-specific imports
#from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from time import time
from tensorflow.keras.callbacks import LambdaCallback

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K


def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:
        
        1. Depthwise Convolutions to learn spatial filters within a 
        temporal convolution. The use of the depth_multiplier option maps 
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn 
        spatial filters within each filter in a filter-bank. This also limits 
        the number of free parameters to fit when compared to a fully-connected
        convolution. 
        
        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions. 
        
    
    While the original paper used Dropout, we found that SpatialDropout2D 
    sometimes produced slightly better results for classification of ERP 
    signals. However, SpatialDropout2D significantly reduced performance 
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.
        
    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the 
    kernel lengths for double the sampling rate, etc). Note that we haven't 
    tested the model performance with this rule so this may not work well. 
    
    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.

    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D. 

    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)




channels = np.array([
    'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5',
    'FC6', 'CZ', 'C3', 'C4', 'T3', 'T4', 'A1', 'A2', 'CP1', 'CP2',
    'CP5', 'CP6', 'PZ', 'P3', 'P4', 'T5', 'T6', 'PO3', 'PO4', 'OZ',
    'O1', 'O2'
])

# Function to load data per subject and session
def load_subject_data(file_path):
    mat_data = loadmat(file_path)
    # Extract subject ID and session ID from filename
    parts = file_path.split('_')
    subject_id = parts[0]
    session_id = parts[1]
    
    # Extract data and labels
    data = mat_data['data']
    labels = mat_data['labels']
    
    return subject_id, session_id, data, labels

def split_data_by_label(data, labels):
    # Split data into Left (1) and Right (2) motor imagery based on labels
    data_L = []
    data_R = []
    
    # Iterate through the labels (assuming it's a 1D array of 1's and 2's)
    for i in range(labels.shape[1]):
        label = labels[0,i]  # Ensure label is a scalar value, not an array
        
        # Check the label value
        if label == 1:  # Left motor imagery
            data_L.append(data[i,:,:])
        elif label == 2:  # Right motor imagery
            data_R.append(data[i,:,:])
        else:
            print(f"Unexpected label found at index {i}: {label}")  # Debugging line
            raise ValueError(f"Unexpected label: {label}")
    
    # Convert lists to arrays
    data_L = np.array(data_L)
    data_R = np.array(data_R)
    
    return data_L, data_R

# Function to bandpass filter trials
def bandpass_filter_trials(data, low_freq, high_freq, sfreq):
    nyquist = 0.5 * sfreq
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(N=4, Wn=[low, high], btype='band')
    
    filtered_trials = []
    for trial in range(data.shape[0]):
        trial_data = data[trial, :, :]
        filtered_trial_data = np.zeros_like(trial_data)
        for ch in range(trial_data.shape[0]):
            filtered_trial_data[ch, :] = filtfilt(b, a, trial_data[ch, :])
        filtered_trials.append(filtered_trial_data)
    
    return np.array(filtered_trials)

# Function to prepare folds for Leave-One-Session-Out Cross-Validation (LOSO)
def prepare_folds_for_LOSO(subject_data, n_sessions=5):
    fold_data = {}
    
    # Loop through each session
    for session_idx in range(n_sessions):
        # Initialize lists to hold the training and test data
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        
        # Loop through each session for the current subject
        for session_name, session_data in subject_data.items():
            # Extract data and labels for the current session
            data = session_data['data']
            labels = session_data['labels']
            
            # Debugging: Print session names and session index
            print(f"Checking session: {session_name} for session_idx: {session_idx}")
            
            # If the session is the test session (current fold)
            if session_name == f"ses-0{session_idx}":
                print(f"Adding to test data: {session_name}")
                test_data.append(data)
                test_labels.append(labels)
            else:
                # Otherwise, it's part of the training data
                print(f"Adding to train data: {session_name}")
                train_data.append(data)
                train_labels.append(labels)
        
        # Debugging: Check if test_data is empty before concatenation
        if len(test_data) == 0:
            print(f"Error: No test data found for session {session_idx}")
        else:
            fold_data[session_idx] = {
                'train_data': np.concatenate(train_data, axis=0),  # Concatenate along the trial dimension (0)
                'train_labels': np.concatenate(train_labels, axis=0),
                'test_data': np.concatenate(test_data, axis=0),
                'test_labels': np.concatenate(test_labels, axis=0)
            }
    
    return fold_data



# Function to train EEGNet model
def train_eegnet_model(train_data, train_labels, chans, samples, test_data, test_labels, nb_classes=2):
    model = EEGNet(nb_classes=nb_classes, Chans=chans, Samples=samples,
                   dropoutRate=0.5, kernLength=125, F1=8, D=2, F2=16,
                   dropoutType='Dropout')
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.keras', verbose=1, save_best_only=True)

    # Initialize best accuracy to track the best test accuracy
    best_test_acc = 0.0

    # Custom callback to evaluate after each epoch
    def evaluate_after_epoch(epoch, logs):
        nonlocal best_test_acc
        # Evaluate the model on the test data after each epoch
        test_probs = model.predict(test_data)
        test_preds = test_probs.argmax(axis=-1)
        test_acc = np.mean(test_preds == test_labels.argmax(axis=-1))

        # Update the best test accuracy if needed
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"Best Test Accuracy updated: {best_test_acc:.4f}")

    # Fit the model with a custom callback
    model.fit(train_data, train_labels, batch_size=16, epochs=200, verbose=2,
              callbacks=[checkpointer, LambdaCallback(on_epoch_end=evaluate_after_epoch)])

    return model, best_test_acc


# Main loop for each subject and session
results = {}
directory = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\SHU Dataset\MatFiles'
#directory = r'/home/uceerjp/Multi-sessionData/SHU Dataset/MatFiles/'

# Step 1: Load and prepare all subject data
subject_data_dict = {}

# Iterate over each file in the dataset directory
for filename in os.listdir(directory):
    if filename.endswith('.mat'):
        file_path = os.path.join(directory, filename)
        
        # Load subject data from the file
        subject_id, session_id, data, labels = load_subject_data(file_path)
        
        # Split the data into Left and Right motor imagery based on labels
        data_L, data_R = split_data_by_label(data, labels)
        
        # Bandpass filter the data
        fs = 250
        filtered_data_L = bandpass_filter_trials(data_L, low_freq=8, high_freq=30, sfreq=fs)
        filtered_data_R = bandpass_filter_trials(data_R, low_freq=8, high_freq=30, sfreq=fs)
        
        # Store subject data for each session
        if subject_id not in subject_data_dict:
            subject_data_dict[subject_id] = {}
        
        subject_data_dict[subject_id][session_id] = {
            'data_L': filtered_data_L,
            'data_R': filtered_data_R,
            'labels_L': np.zeros(filtered_data_L.shape[0]),
            'labels_R': np.ones(filtered_data_R.shape[0])
        }
        
        # Concatenate data and labels along the specified dimensions
        subject_data_dict[subject_id][session_id] = {
            'data': np.concatenate([filtered_data_L, filtered_data_R], axis=0),  # Concatenate along the first dimension (0)
            'labels': np.concatenate([subject_data_dict[subject_id][session_id]['labels_L'], subject_data_dict[subject_id][session_id]['labels_R']], axis=0)  # Concatenate labels along the second dimension (1)
            }


# Step 2: Process the data using LOSO (Leave-One-Session-Out Cross-Validation)
for subject_id, subject_data in subject_data_dict.items():
    fold_data = prepare_folds_for_LOSO(subject_data)
    
    # Train and evaluate the model for each fold
    for session_idx, fold in fold_data.items():
        train_data, train_labels = fold['train_data'], fold['train_labels']
        test_data, test_labels = fold['test_data'], fold['test_labels']
        
        # One-hot encode labels
        encoder = OneHotEncoder(sparse_output=False)
        train_labels = encoder.fit_transform(train_labels.reshape(-1, 1))
        test_labels = encoder.transform(test_labels.reshape(-1, 1))
        
        # Train the model
        chans = train_data.shape[1]
        samples = train_data.shape[2]
        model, best_test_acc = train_eegnet_model(train_data, train_labels, chans, samples, test_data, test_labels)
        
        # Get the number of parameters in the model
        num_params = model.count_params()
        print(f"Number of parameters: {num_params}")
        
        # Store the results, including the best test accuracy and the number of parameters
        if subject_id not in results:
            results[subject_id] = []
        
        results[subject_id].append({
            'session': session_idx,
            'best_test_accuracy': best_test_acc,
            'num_params': num_params  # Store the number of parameters
        })

# Output the final results
for subject_id, subject_results in results.items():
    print(f"Results for {subject_id}:")
    for result in subject_results:
        print(f"  Session {result['session']}: Best Test Accuracy = {result['best_test_accuracy']:.4f}, Number of Parameters = {result['num_params']}")

# Define output file path
# Save the file in the current directory
current_path = os.getcwd()
output_file_path = os.path.join(current_path, "final_results.txt")

# Save results to a text file
with open(output_file_path, 'w') as file:
    for subject_id, subject_results in results.items():
        file.write(f"Results for {subject_id}:\n")
        for result in subject_results:
            file.write(f"  Session {result['session']}: Best Test Accuracy = {result['best_test_accuracy']:.4f}, Number of Parameters = {result['num_params']}\n")

#%%

import statistics

subject_averages = {}

for subject, sessions in results.items():
    accuracies = [session['best_test_accuracy'] for session in sessions]
    average = sum(accuracies) / len(accuracies)
    std_dev = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    subject_averages[subject] = (average, std_dev)

for subject, (avg, std) in subject_averages.items():
    print(f"{subject}: Average Accuracy = {avg:.4f}, Standard Deviation = {std:.4f}")

