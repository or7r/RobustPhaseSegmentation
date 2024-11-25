import numpy as np
import os

def load(path):
    return np.load(path, allow_pickle=True).item()

def save(path, labels):
    np.save(path, labels, allow_pickle=True)



def downsample(file_path, downsample_rate=2):
    # This function loads the labels from the file_path, 
    # Then it selects half of the keys and saves the new labels to a new file
    # but keep the order of the values
    # The function then saves the new labels to a new file
    labels = load(file_path)
    folder_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)

    for key in labels.keys():
        labels[key] = labels[key][::downsample_rate]

    
    downsampled_labels_path = os.path.join(folder_name, base_name + f'_{downsample_rate}-real-downsampled.npy')
    save(downsampled_labels_path, labels)

if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print('Usage: python downsample_labels.py <path>')
        sys.exit(1)

    orig_path = sys.argv[1]
    downsample(orig_path)   