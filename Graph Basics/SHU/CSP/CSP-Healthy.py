import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP

# Define the directory containing your .mat files
directory = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\SHU Dataset\MatFiles'

# Load data and split into subjects
def load_data(directory):
    data_by_subject = {}

    for filename in os.listdir(directory):
        if filename.endswith('.mat'):
            file_path = os.path.join(directory, filename)
            mat_data = loadmat(file_path)
            parts = filename.split('_')
            subject_id = parts[0]
            session_id = parts[1]

            if subject_id not in data_by_subject:
                data_by_subject[subject_id] = {
                    'data': {},
                    'labels': {}
                }

            data = mat_data['data']
            labels = mat_data['labels']

            data_by_subject[subject_id]['data'][session_id] = data
            data_by_subject[subject_id]['labels'][session_id] = labels

    return data_by_subject

data_by_subject = load_data(directory)

def bandpass_filter(data, low_freq, high_freq, sfreq):
    nyquist = 0.5 * sfreq
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(N=4, Wn=[low, high], btype='band')

    filtered_data = []
    for trial in data:
        filtered_trial = np.zeros_like(trial)
        for ch in range(trial.shape[0]):
            filtered_trial[ch, :] = filtfilt(b, a, trial[ch, :])
        filtered_data.append(filtered_trial)

    return np.array(filtered_data, dtype=np.float64)


def leave_one_session_cv(subject_data, subject_labels, sfreq, low_freq, high_freq):
    sessions = list(subject_data.keys())
    results = []

    for test_session in sessions:
        train_data = []
        train_labels = []

        for session in sessions:
            if session != test_session:
                train_data.append(subject_data[session])
                train_labels.append(subject_labels[session].squeeze())  # Ensure 1D labels

        # Concatenate training data and labels
        train_data = np.concatenate(train_data, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)  # Use axis=0 for 1D labels

        test_data = subject_data[test_session]
        test_labels = subject_labels[test_session].squeeze()  # Ensure 1D labels

        # Debug shapes
        print(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")
        print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")

        # Filter the data
        train_data = bandpass_filter(train_data, low_freq, high_freq, sfreq)
        test_data = bandpass_filter(test_data, low_freq, high_freq, sfreq)

        # Train and test CSP + SVM
        csp = CSP(n_components=10, log=True, norm_trace=True)
        clf = make_pipeline(csp, StandardScaler(), SVC(kernel='linear'))

        clf.fit(train_data, train_labels)
        predictions = clf.predict(test_data)
        accuracy = accuracy_score(test_labels, predictions)

        results.append((test_session, accuracy))

    return results


# Process each subject and perform cross-validation
sfreq = 250
low_freq = 8
high_freq = 30

# Initialize a dictionary to store results for each subject
all_subject_results = {}

for subject_id, subject_info in data_by_subject.items():
    subject_data = subject_info['data']
    subject_labels = subject_info['labels']

    # Run cross-validation for the current subject
    results = leave_one_session_cv(subject_data, subject_labels, sfreq, low_freq, high_freq)

    # Store results in a dictionary
    subject_results = [{"test_session": session, "accuracy": accuracy} for session, accuracy in results]
    all_subject_results[subject_id] = subject_results

    # Print results for the current subject
    print(f"Results for {subject_id}:")
    for session, accuracy in results:
        print(f"  Test Session: {session}, Accuracy: {accuracy:.2f}")
        
#%%

# Compute and print average accuracy and standard deviation per subject
for subject_id, results in all_subject_results.items():
    # Extract accuracy values
    accuracies = [entry['accuracy'] for entry in results]
    
    # Calculate mean and standard deviation
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    # Print results
    print(f"Subject: {subject_id}")
    print(f"  Average Accuracy: {mean_accuracy:.4f}")
    print(f"  Standard Deviation: {std_accuracy:.4f}")
    print()
    