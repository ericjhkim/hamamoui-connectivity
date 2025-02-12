import h5py
import matplotlib.pyplot as plt
import os
from datetime import datetime

#%% Plotting and Visualization
def plot_data(N, P_THRESH, phis, data):
    plt.plot(phis, data, marker='o')
    plt.xlabel('Connectivity Factor (phi)')
    plt.ylabel('Pr')
    plt.title(f'Matching Frequency, (n={N}, p_thresh={P_THRESH})')
    plt.grid(True)
    plt.show()

def plot_multi_data(N, P_THRESH, phis, data):
    # Create the plot
    plt.figure(figsize=(10, 6))

    for i in range(len(data)):
        plt.plot(phis, data[i], label=f'N={N[i]}, P_THRESH={P_THRESH[i]}')
        plt.scatter(phis, data[i], marker='o', color='black', s=15, zorder=2)

    # Add labels, legend, and grid
    plt.title(f'Matching Frequency')
    plt.xlabel('Connectivity Factor (phi)')
    plt.ylabel('Pr')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()

#%% Data Management
def load_data(data_path):
    with h5py.File(data_path, 'r') as f:
        N = f['N'][()]
        P_THRESH = f['P_THRESH'][()]
        phis = f['phis'][:]
        matching_frequencies = f['matching_frequencies'][:]
        pr = f['pr'][:]
    return N, P_THRESH, phis, matching_frequencies, pr

def get_latest_h5_file(folder_path):
    latest_file = None
    latest_time = None

    # List all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file matches the expected format
        if filename.startswith("data_") and filename.endswith(".h5"):
            try:
                # Extract the timestamp from the filename
                timestamp_str = filename[5:-3]  # Remove 'data_' prefix and '.h5' suffix
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                # Update the latest file if this one is newer
                if latest_time is None or timestamp > latest_time:
                    latest_time = timestamp
                    latest_file = os.path.join(folder_path, filename)

            except ValueError:
                # Skip files that don't match the expected timestamp format
                continue

    return latest_file